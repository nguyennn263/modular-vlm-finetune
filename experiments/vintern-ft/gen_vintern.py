"""Generate predictions from a (merged) Vintern-1B model on an AutoViVQA split.

Generation ONLY — no metric computation here, so the kernel stays in the
InternVL-training environment (transformers 4.47) with no extra deps. Writes
`text_predictions_epoch_1.json` in the bridge-pipeline format; score it locally
with experiments/vintern-ft/score_local.py (same metric stack as Table 1).

- Loads via InternVLChatModel (the Vintern repo's class), same as merge_lora.py.
- Greedy: do_sample=False, num_beams=1, max_new_tokens=64 (matches bridge eval).
- Tiling: max_num tiles (default 6 = the fine-tune's max_dynamic_patch).

    python experiments/vintern-ft/gen_vintern.py \
        --model-path work_dirs/vintern_lora_merge --data .../autovivqa_val.jsonl \
        --images-dir /kaggle/input/auto-vqabest/preprocessed_images --out /kaggle/working/out/val
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

sys.path.append("/tmp/wk/Vintern/internvl_chat")
from internvl.model.internvl_chat import InternVLChatModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(sz):
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def closest_ratio(ar, ratios, w, h, sz):
    best_d, best = float("inf"), (1, 1)
    area = w * h
    for r in ratios:
        d = abs(ar - r[0] / r[1])
        if d < best_d:
            best_d, best = d, r
        elif d == best_d and area > 0.5 * sz * sz * r[0] * r[1]:
            best = r
    return best


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    tr = closest_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    blocks = tr[0] * tr[1]
    resized = image.resize((tw, th))
    tiles = [resized.crop((
        (i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
        ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size))
        for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(path, sz=448, max_num=6):
    im = Image.open(path).convert("RGB")
    tf = build_transform(sz)
    return torch.stack([tf(t) for t in dynamic_preprocess(im, image_size=sz, max_num=max_num)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-num", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    outp = Path(a.out) / "results"
    outp.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

    gcfg = dict(max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    samples = []
    for i, r in enumerate(rows):
        pv = load_image(os.path.join(a.images_dir, r["image"]), max_num=a.max_num).to(torch.bfloat16).cuda()
        q = r["conversations"][0]["value"]  # "<image>\n{question}"
        try:
            pred = model.chat(tok, pv, q, gcfg)
        except Exception as e:  # noqa
            pred = f"[gen-error: {str(e)[:100]}]"
        samples.append({"index": i, "question": q.replace("<image>\n", ""),
                        "prediction": pred.strip() if isinstance(pred, str) else str(pred),
                        "ground_truths": r["all_answers"]})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    pf = outp / "text_predictions_epoch_1.json"
    pf.write_text(json.dumps({"epoch": 1, "split": a.split, "samples": samples}, ensure_ascii=False, indent=1))

    # quick inline sanity F1 (set-overlap best-over-refs) — real scoring is local
    import re

    def toks(s):
        return set(re.sub(r"[^\w\s]", " ", s.lower()).split())
    f1s = []
    for s in samples:
        p = toks(s["prediction"])
        best = 0.0
        for g in s["ground_truths"]:
            gg = toks(g)
            if p and gg:
                ov = len(p & gg)
                pr, rc = ov / len(p), ov / len(gg)
                if pr + rc:
                    best = max(best, 2 * pr * rc / (pr + rc))
        f1s.append(best)
    print(f"[{a.split}] n={len(samples)}  quick-F1≈{100*sum(f1s)/len(f1s):.2f}  -> {pf}")


if __name__ == "__main__":
    main()
