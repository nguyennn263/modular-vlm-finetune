"""Training-free hybrid: our trained Multi-Token + decoder-LoRA checkpoint, plus Vintern's
own mlp1 tokens (pixel-shuffle + mlp1 on the same 336px image, 144 tokens), chosen per
question. Nothing is trained: ViT, mlp1, bridge and LoRA are used as they are.

Selection (qsim + IQR): score each mlp1 token by its max cosine with the question's word
embeddings (both mean-centred), keep the tokens whose score is a Tukey outlier within
the image, score > Q3 + 1.5 * IQR. k varies per question; k = 0 reproduces the
checkpoint exactly.

Configs
  ours8        : the checkpoint as trained (8 global tokens prefixed to the prompt)
  dyn          : ours8 + the qsim-IQR tokens in Vintern's own image slot (<img>...</img>)
  rand         : ours8 + the same number of random tokens, same slot
  full         : ours8 + all 144 tokens, same slot
  full_prefix  : ours8 + all 144 tokens appended right after the 8 global tokens

    python experiments/token-select/hybrid_plugin.py --checkpoint l3ep-s42.pt \
        --ood-data <dir>/ours.jsonl --ood-images <dir>/images --n-ood 20 --n-val 20 --out out/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
SYSTEM = ("Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi là Vintern, "
          "được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ nhân tạo hữu ích và không gây hại.")
CONFIGS = ["ours8", "dyn", "rand", "full", "full_prefix"]


def prompt(question: str, n_img: int) -> str:
    """The trainer's prompt (trainer._build_prompt_text); with n_img > 0 the literal
    <image> becomes Vintern's image slot holding n_img context tokens."""
    image = "<image>" if n_img == 0 else "<img>" + "<IMG_CONTEXT>" * n_img + "</img>"
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{image}\n{question}<|im_end|>\n<|im_start|>assistant\n")


def iqr_keep(scores: torch.Tensor) -> torch.Tensor:
    q1, q3 = torch.quantile(scores, torch.tensor([0.25, 0.75], device=scores.device))
    return (scores > q3 + 1.5 * (q3 - q1)).nonzero().flatten()


class Plugin:
    def __init__(self, base, model, tok, device):
        self.base, self.m, self.tok, self.dev = base, model, tok, device
        self.dtype = next(model.vision_model.parameters()).dtype
        self.ctx_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        table = (model.language_model.get_base_model() if getattr(model, "lora_enabled", False)
                 else model.language_model).model.embed_tokens.weight
        self.vocab_mean = table.float().mean(0)

    @torch.no_grad()
    def encode(self, image_path: str, question: str):
        from src.data.collator import load_image
        pv = load_image(image_path, size=(336, 336)).unsqueeze(0).to(self.dev, self.dtype)
        hid = self.m.vision_model(pv).last_hidden_state                       # (1, 577, 1024)
        glob = self.m.bridge(hid[:, 0])                                        # (1, 8, 896)
        patches = hid[:, 1:]
        side = int(patches.shape[1] ** 0.5)
        grid = self.base.pixel_shuffle(patches.reshape(1, side, side, -1), scale_factor=self.base.downsample_ratio)
        local = self.base.mlp1(grid.reshape(1, -1, grid.shape[-1]))            # (1, 144, 896)
        ids = self.tok(question, add_special_tokens=False, return_tensors="pt").input_ids.to(self.dev)
        q = F.normalize(self.m.embed_text(ids)[0].float() - self.vocab_mean, dim=-1)
        v = local[0].float()
        v = F.normalize(v - v.mean(0), dim=-1)
        scores = (v @ q.T).max(-1).values                                      # (144,)
        return glob, local, scores

    @torch.no_grad()
    def generate(self, question: str, prefix: torch.Tensor, slot: torch.Tensor | None) -> str:
        n_img = 0 if slot is None else slot.shape[1]
        enc = self.tok(prompt(question, n_img), return_tensors="pt")
        ids, mask = enc.input_ids.to(self.dev), enc.attention_mask.to(self.dev)
        text = self.m.embed_text(ids).to(self.dtype)
        if n_img:
            text[0, ids[0] == self.ctx_id] = slot[0].to(self.dtype)
        emb = torch.cat([prefix.to(self.dtype), text], 1)
        att = torch.cat([torch.ones(1, prefix.shape[1], device=self.dev, dtype=mask.dtype), mask], 1)
        out = self.m.language_model.generate(
            inputs_embeds=emb, attention_mask=att, max_new_tokens=50, do_sample=False, num_beams=1,
            pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True).strip() or "[Empty output]"

    def run(self, image_path: str, question: str, seed: int) -> dict:
        glob, local, scores = self.encode(image_path, question)
        keep = iqr_keep(scores)
        k = int(keep.numel())
        rnd = torch.randperm(local.shape[1], generator=torch.Generator().manual_seed(seed))[:k].sort().values
        ours8 = self.generate(question, glob, None)
        preds = {
            "ours8": ours8,
            "dyn": ours8 if k == 0 else self.generate(question, glob, local[:, keep]),
            "rand": ours8 if k == 0 else self.generate(question, glob, local[:, rnd.to(self.dev)]),
            "full": self.generate(question, glob, local),
            "full_prefix": self.generate(question, torch.cat([glob, local], 1), None),
        }
        return {"k": k, "kept": keep.tolist(), "preds": preds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ood-data", required=True)
    ap.add_argument("--ood-images", required=True)
    ap.add_argument("--n-ood", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    from src.training import create_finetune_model
    from src.data.split import load_split
    from metrics.vqa_metrics import score_answers

    dev = torch.device("cuda")
    base = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
                                     trust_remote_code=True).eval()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = create_finetune_model(base, bridge_type="multi_token", bridge_config={"num_tokens": 8},
                                  lora={} if "lora_state" in ckpt else None)
    model.bridge.load_state_dict(ckpt.get("bridge_state", ckpt))
    if "lora_state" in ckpt:
        model.load_lora_state_dict(ckpt["lora_state"])
    model.to(dev).eval()
    base.mlp1.to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    plug = Plugin(base, model, tok, dev)

    ood = [json.loads(l) for l in open(a.ood_data, encoding="utf-8")][: a.n_ood]
    sets = {
        "vitextvqa": [(str(Path(a.ood_images) / r["image_name"]), r["question"], list(r["answers"]))
                      for r in ood],
        "val": [(s.image_path, s.question, list(s.answers))
                for s in load_split("val", a.split_dir)[: a.n_val]],
    }
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rows in sets.items():
        recs = []
        for i, (img, q, gts) in enumerate(rows):
            r = plug.run(img, q, seed=i)
            recs.append({"image": Path(img).name, "question": q, "ground_truths": gts, **r})
            print(f"[{name} {i}] k={r['k']:3d} | {q[:60]} | GT {gts[:1]}", flush=True)
            for c in CONFIGS:
                print(f"     {c:11s} {r['preds'][c][:80]}", flush=True)
        summary = {}
        for c in CONFIGS:
            avg, _ = score_answers([x["preds"][c] for x in recs], [x["ground_truths"] for x in recs])
            summary[c] = {m: 100 * avg[m] for m in ("accuracy", "f1", "cider")}
        ks = [x["k"] for x in recs]
        report[name] = {"n": len(recs), "k_mean": sum(ks) / len(ks), "k": ks, "summary": summary}
        (out_dir / f"{name}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        print(f"== {name}: mean k {report[name]['k_mean']:.1f}", flush=True)
        for c in CONFIGS:
            print(f"   {c:11s} F1 {summary[c]['f1']:6.2f}  CIDEr {summary[c]['cider']:7.2f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
