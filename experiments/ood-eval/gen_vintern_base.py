"""Zero-shot Vintern-1B-v3_5 ("Vintern goc") generation over an OOD dataset built
by build_ood_data.py. Loads straight from the HF Hub via trust_remote_code --
no Vintern GitHub clone needed (that's only required for the *training* code;
.chat() inference is fully self-contained in the HF repo's remote code).

Tiling: max_num=6 (matches this project's own fine-tune-cookbook max_dynamic_patch
and keeps compute reasonable; Vintern's base config allows up to 12, but 6 is
InternVL's own commonly-used default for .chat() usage examples). Greedy decode,
max_new_tokens=64 -- same generation settings used throughout this project's own
bridge/Vintern-FT eval, so the comparison isn't confounded by decoding params.

Writes results/text_predictions_epoch_1.json in the same bridge-pipeline format
used everywhere else in this repo (score with experiments/vintern-ft/score_local.py).

    python gen_vintern_base.py --data /kaggle/working/data/vitextvqa/internvl.jsonl \
        --images-dir /kaggle/working/data/vitextvqa/images --out /kaggle/working/out/vitextvqa/vintern_base
"""
import argparse
import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-num", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    outp = os.path.join(a.out, "results")
    os.makedirs(outp, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()

    gcfg = dict(max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    samples = []
    for i, r in enumerate(rows):
        pv = load_image(os.path.join(a.images_dir, r["image"]), max_num=a.max_num).to(torch.bfloat16).cuda()
        q = r["conversations"][0]["value"]
        try:
            pred = model.chat(tok, pv, q, gcfg)
        except Exception as e:  # noqa
            pred = f"[gen-error: {str(e)[:100]}]"
        samples.append({"index": i, "question": q.replace("<image>\n", ""),
                        "prediction": pred.strip() if isinstance(pred, str) else str(pred),
                        "ground_truths": r["all_answers"]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    pf = os.path.join(outp, "text_predictions_epoch_1.json")
    json.dump({"epoch": 1, "model": MODEL_NAME, "max_num": a.max_num, "samples": samples},
               open(pf, "w"), ensure_ascii=False, indent=1)
    print(f"n={len(samples)} -> {pf}")


if __name__ == "__main__":
    main()
