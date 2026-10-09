"""Convert our grouped leak-free split -> InternVL chat SFT format.

Follows the official Vintern fine-tune cookbook
(experiments/vintern-ft/reference/official_vintern_finetune_colab.ipynb, cells 26/30):

    {"id", "image", "width", "height",
     "conversations": [{"from":"human","value":"<image>\\n{q}"},
                       {"from":"gpt","value":"{a}"}]}

Answer target = answers[0] (the first reference) to match the bridge pipeline's
answer-sampling=first, which was the best training target in our ablation.

Usage:
    python experiments/vintern-ft/build_data.py \
        --splits-dir data/splits --images-dir data/raw/images \
        --out-dir experiments/vintern-ft/data
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROMPT_SUFFIX = ""  # keep the raw question; AutoViVQA questions are self-contained


def build_split(split, splits_dir, images_dir, out_dir):
    src = Path(splits_dir) / f"{split}.jsonl"
    dst = Path(out_dir) / f"autovivqa_{split}.jsonl"
    rows = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=split)):
            d = json.loads(line)
            img_rel = d["image_name"]
            img_abs = Path(images_dir) / img_rel
            with Image.open(img_abs) as im:
                w, h = im.size
            answer = d["answers"][0].strip()
            rows.append({
                "id": i,
                "image": img_rel,  # meta["root"] is prepended by InternVL loader
                "width": w,
                "height": h,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["question"].strip() + PROMPT_SUFFIX},
                    {"from": "gpt", "value": answer},
                ],
                # carried for eval only (InternVL ignores unknown keys on train):
                "all_answers": d["answers"],
                "category": d.get("category"),
            })
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dst, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--images-dir", default="data/raw/images")
    ap.add_argument("--out-dir", default="experiments/vintern-ft/data")
    ap.add_argument("--meta-image-root", default="/kaggle/input/autovivqa-images/images",
                    help="images dir as seen at TRAIN time (Kaggle mount path)")
    args = ap.parse_args()

    meta = {}
    for split in ["train", "val", "test"]:
        dst, n = build_split(split, args.splits_dir, args.images_dir, args.out_dir)
        print(f"  {split}: {n} -> {dst}")
        if split == "train":
            meta["autovivqa-train"] = {
                "root": args.meta_image_root,
                "annotation": str(dst),
                "data_augment": False,
                "repeat_time": 1,
                "length": n,
            }
    meta_path = Path(args.out_dir) / "meta_autovivqa.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  meta -> {meta_path}")


if __name__ == "__main__":
    main()
