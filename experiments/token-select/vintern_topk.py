"""Training-free check: how few of Vintern's own mlp1 tokens does a text-heavy OOD set
need, and does the choice of tokens matter? Vintern-1B-v3_5 as released, one 336px
tile (24x24 patches -> pixel-shuffle -> mlp1 -> 144 tokens, the same input size as
every bridge in the paper). For each scorer, keep the top-k of the 144 tokens (in
their original spatial order) and generate with only those.

Scorers
  random : uniform random subset (floor)
  cls    : cosine of each ViT patch with the CLS token, averaged over the 2x2 group
           that pixel-shuffle merges into one token (question-agnostic, PruMerge-like)
  qsim   : max cosine between a token and the question's word embeddings in Qwen's
           input space, both mean-centred (question-aware, no LLM pass)
  attn   : attention from the last prompt token to each image token at decoder
           layer 2, averaged over heads, one pass with all 144 tokens (FastV-like)

    python experiments/token-select/vintern_topk.py --data <dir>/internvl.jsonl \
        --images-dir <dir>/images --out /kaggle/working/out --ks 6,14,32
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from metrics.vqa_metrics import score_answers  # noqa: E402

MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
SIZE = 336
ATTN_LAYER = 2
IMG_CONTEXT, IMG_START, IMG_END = "<IMG_CONTEXT>", "<img>", "</img>"
TRANSFORM = T.Compose([
    T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
    T.Resize((SIZE, SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


class Selector:
    """Computes Vintern's 144 mlp1 tokens once per image and scores them."""

    def __init__(self, model, tok):
        self.m, self.tok = model, tok
        self.emb = model.language_model.get_input_embeddings()
        self.vocab_mean = self.emb.weight.float().mean(0)
        self.conv = sys.modules[type(model).__module__].get_conv_template

    @torch.no_grad()
    def encode(self, pv):
        hid = self.m.vision_model(pixel_values=pv, output_hidden_states=False, return_dict=True).last_hidden_state
        cls, patches = hid[:, 0], hid[:, 1:]
        side = int(patches.shape[1] ** 0.5)
        grid = self.m.pixel_shuffle(patches.reshape(1, side, side, -1), scale_factor=self.m.downsample_ratio)
        tokens = self.m.mlp1(grid.reshape(1, -1, grid.shape[-1]))               # (1, 144, 896)
        cos = F.cosine_similarity(patches.float(), cls.float()[:, None], dim=-1)  # (1, 576)
        cls_score = self.m.pixel_shuffle(cos.reshape(1, side, side, 1), scale_factor=self.m.downsample_ratio)
        return tokens, cls_score.mean(-1).reshape(-1)                           # same order as tokens

    def prompt(self, question, n_img):
        t = self.conv(self.m.template)
        t.system_message = self.m.system_message
        t.append_message(t.roles[0], "<image>\n" + question)
        t.append_message(t.roles[1], None)
        return t.get_prompt().replace("<image>", IMG_START + IMG_CONTEXT * n_img + IMG_END, 1), t

    @torch.no_grad()
    def qsim(self, tokens, question):
        ids = self.tok(question, add_special_tokens=False, return_tensors="pt").input_ids.to(tokens.device)
        q = F.normalize(self.emb(ids)[0].float() - self.vocab_mean, dim=-1)       # (L, 896)
        v = tokens[0].float()
        v = F.normalize(v - v.mean(0), dim=-1)                                  # (144, 896)
        return (v @ q.T).max(-1).values

    @torch.no_grad()
    def attn(self, tokens, question):
        query, _ = self.prompt(question, tokens.shape[1])
        ids = self.tok(query, return_tensors="pt").input_ids.to(tokens.device)
        emb = self.emb(ids)
        img = ids[0] == self.m.img_context_token_id
        emb[0, img] = tokens[0].to(emb.dtype)
        out = self.m.language_model(inputs_embeds=emb, output_attentions=True, return_dict=True)
        assert out.attentions is not None, "decoder returned no attentions (flash-attn?)"
        return out.attentions[ATTN_LAYER][0, :, -1].float().mean(0)[img]        # (144,)

    @torch.no_grad()
    def generate(self, feats, question, gcfg):
        query, t = self.prompt(question, feats.shape[1])
        enc = self.tok(query, return_tensors="pt").to(feats.device)
        cfg = dict(gcfg, eos_token_id=self.tok.convert_tokens_to_ids(t.sep.strip()))
        out = self.m.generate(pixel_values=feats, visual_features=feats, input_ids=enc.input_ids,
                              attention_mask=enc.attention_mask, **cfg)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].split(t.sep.strip())[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="6,14,32")
    ap.add_argument("--scorers", default="random,cls,qsim,attn")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    scorers = a.scorers.split(",")
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    # eager attention everywhere so the attn scorer can read attention weights
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=False, trust_remote_code=True).eval().cuda()
    model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)
    sel = Selector(model, tok)
    gcfg = dict(max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = rows[: a.limit] if a.limit else rows
    configs = ["full144"] + [f"{s}_k{k}" for s in scorers for k in ks]
    preds = {c: [] for c in configs}
    kept = {c: [] for c in configs if c != "full144"}   # kept token ids on the 12x12 grid, row-major
    gts, t0 = [], time.time()
    for i, r in enumerate(rows):
        q = r["conversations"][0]["value"].replace("<image>\n", "")
        gts.append(r["all_answers"])
        pv = TRANSFORM(Image.open(os.path.join(a.images_dir, r["image"]))).unsqueeze(0).to(torch.bfloat16).cuda()
        tokens, cls_score = sel.encode(pv)
        n = tokens.shape[1]
        scores = {"cls": cls_score, "random": torch.rand(n, generator=torch.Generator().manual_seed(i))}
        if "qsim" in scorers:
            scores["qsim"] = sel.qsim(tokens, q)
        if "attn" in scorers:
            scores["attn"] = sel.attn(tokens, q)
        preds["full144"].append(sel.generate(tokens, q, gcfg))
        for s in scorers:
            for k in ks:
                idx = scores[s].float().cpu().topk(min(k, n)).indices.sort().values
                kept[f"{s}_k{k}"].append(idx.tolist())
                preds[f"{s}_k{k}"].append(sel.generate(tokens[:, idx.to(tokens.device)], q, gcfg))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f}s/sample", flush=True)

    summary = {}
    for c in configs:
        avg, _ = score_answers(preds[c], gts)
        summary[c] = {m: 100 * avg[m] for m in ("accuracy", "f1", "cider")}
        print(f"{c:12s} F1 {summary[c]['f1']:6.2f}  CIDEr {summary[c]['cider']:7.2f}", flush=True)
    json.dump({"n": len(rows), "size": SIZE, "attn_layer": ATTN_LAYER, "summary": summary},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump({"images": [r["image"] for r in rows],
               "questions": [r["conversations"][0]["value"] for r in rows], "ground_truths": gts,
               "predictions": preds, "kept": kept}, open(os.path.join(a.out, "predictions.json"), "w"),
              ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
