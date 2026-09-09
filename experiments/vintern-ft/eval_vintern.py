"""Evaluate a (merged) Vintern-1B model on an AutoViVQA split with OUR metric
conventions, so the number is directly comparable to the bridge pipeline's
Table 1 (in-house 8-metric) and the corpus rows.

- Generation: greedy (do_sample=False, num_beams=1), max_new_tokens=64 — matches
  the bridge eval, NOT the cookbook's beam-3 inference demo.
- Tiling at eval: max_num tiles (default 6 = the fine-tune's max_dynamic_patch).
- In-house metrics via metrics.vqa_metrics (same classes the trainer uses).
- Writes text_predictions_epoch_1.json in the bridge pipeline's format so
  scripts/rescore_corpus.py can score the corpus CIDEr-D / BLEU-4 / ROUGE-L.

    python experiments/vintern-ft/eval_vintern.py \
        --model-path work_dirs/vintern_1b_v3_5_autovivqa_lora_merge \
        --split val --data experiments/vintern-ft/data/autovivqa_val.jsonl \
        --images-dir /kaggle/input/autovivqa-images/images \
        --out work_dirs/eval/val
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        tar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    ow, oh = image.size
    ar = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    tar = find_closest_aspect_ratio(ar, target_ratios, ow, oh, image_size)
    tw, th = image_size * tar[0], image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = ((i % (tw // image_size)) * image_size,
               (i // (tw // image_size)) * image_size,
               ((i % (tw // image_size)) + 1) * image_size,
               ((i // (tw // image_size)) + 1) * image_size)
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image(path, input_size=448, max_num=6):
    image = Image.open(path).convert("RGB")
    tf = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([tf(t) for t in tiles])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-num", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N only")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, use_flash_attn=False,
    ).eval().cuda()

    gen_cfg = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    samples = []
    for i, r in enumerate(rows):
        img_path = os.path.join(args.images_dir, r["image"])
        pixel_values = load_image(img_path, max_num=args.max_num).to(torch.bfloat16).cuda()
        question = r["conversations"][0]["value"]  # already "<image>\n{q}"
        try:
            pred = model.chat(tok, pixel_values, question, gen_cfg)
        except Exception as e:  # noqa
            pred = f"[gen-error: {str(e)[:80]}]"
        samples.append({
            "index": i,
            "question": question.replace("<image>\n", ""),
            "prediction": pred.strip(),
            "ground_truths": r["all_answers"],
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    pred_path = out_dir / "results" / "text_predictions_epoch_1.json"
    pred_path.write_text(json.dumps({"epoch": 1, "samples": samples}, ensure_ascii=False, indent=1))
    print(f"predictions -> {pred_path}")

    # ---- in-house metrics (same classes as src/training/trainer.py) ----
    from metrics.vqa_metrics import (
        BLEUScore, METEORScore, ROUGEScore, CIDErScore,
        PrecisionRecallF1, ExactMatchAccuracy,
    )
    gens = [s["prediction"] for s in samples]
    gts = [s["ground_truths"] for s in samples]

    res = {}
    prf = PrecisionRecallF1()
    prf.update(gens, gts)
    pm = prf.compute().metadata
    res["precision"], res["recall"], res["f1"] = (
        float(pm["precision"]), float(pm["recall"]), float(pm["f1"]))

    for name, m in [
        ("bleu", BLEUScore(n_gram=4)), ("meteor", METEORScore()),
        ("rouge_l", ROUGEScore(rouge_type="rougeL")), ("cider", CIDErScore(n_gram=4)),
        ("exact_match", ExactMatchAccuracy(normalize=True)),
    ]:
        m.update(gens, gts)
        res[name] = float(m.compute().value)

    acc = np.mean([
        1.0 if s["prediction"].lower().strip() in {g.lower().strip() for g in s["ground_truths"]}
        else 0.0 for s in samples])
    res["accuracy"] = float(acc)
    res["n"] = len(samples)
    res["split"] = args.split
    res["scale_x100"] = {k: round(v * 100, 2) for k, v in res.items()
                         if isinstance(v, float) and k != "n"}

    metrics_path = out_dir / "results" / "inhouse_metrics.json"
    metrics_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res["scale_x100"], ensure_ascii=False, indent=2))
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
