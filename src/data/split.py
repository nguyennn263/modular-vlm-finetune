"""Build the 70/15/15 train/val/test split (final-plan P2).

Grouped by ``image_id`` (an image never spans two splits) and stratified by the
image's dominant ``category``. Deterministic given ``--seed``.

    python -m src.data.split
    python -m src.data.split --ratios 0.7 0.15 0.15 --seed 42

Reads ``data/labeled.parquet`` (produced by ``src.data.labeled_table``) and writes
``data/splits/{train,val,test}.jsonl`` — one JSON object per QA row.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.config.loader import repo_root

SPLITS = ("train", "val", "test")


def assign(labeled: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.DataFrame:
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    rng = random.Random(seed)

    # One row per image with its dominant category.
    dominant = (
        labeled.groupby("image_id")["category"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("strat")
        .reset_index()
    )

    image_split: dict[int, str] = {}
    for strat, grp in dominant.groupby("strat"):
        ids = list(grp["image_id"])
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, image_id in enumerate(ids):
            if i < n_train:
                image_split[image_id] = "train"
            elif i < n_train + n_val:
                image_split[image_id] = "val"
            else:
                image_split[image_id] = "test"

    out = labeled.copy()
    out["split"] = out["image_id"].map(image_split)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data.split", description=__doc__)
    p.add_argument("--labeled", default="data/labeled.parquet")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    root = repo_root()
    labeled = pd.read_parquet(root / args.labeled)
    tagged = assign(labeled, tuple(args.ratios), args.seed)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["image_id", "image_path", "question", "answers", "category", "reason_depth"]
    for name in SPLITS:
        rows = tagged[tagged["split"] == name][keep]
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for rec in rows.to_dict("records"):
                rec["answers"] = list(rec["answers"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        imgs = tagged[tagged["split"] == name]["image_id"].nunique()
        print(f"[{name:5}] {len(rows):6d} QA   {imgs:6d} images")

    # Leak check.
    per_image_splits = tagged.groupby("image_id")["split"].nunique()
    assert (per_image_splits == 1).all(), "image leaked across splits"
    print(f"[ok] no image leakage; seed={args.seed} ratios={args.ratios}")


if __name__ == "__main__":
    main()
