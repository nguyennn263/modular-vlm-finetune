"""Build the labelled training table (final-plan P0).

Joins the reasoning-type labels in ``final_vqa_dataset.json`` onto the
quality-annotated ``evaluate_60k_data_balanced_preprocessed.csv`` by
``(img_id, question)`` and keeps the 8 canonical ``category`` classes.

    python -m src.data.labeled_table
    python -m src.data.labeled_table --out data/labeled.parquet

Output columns: image_id, image_path, question, answers (list[str]),
category, reason_depth, plus the quality metadata columns from the CSV.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

from src.config.loader import load_config, repo_root

# Vietnamese label -> canonical code. Whitespace is normalised before lookup.
CANONICAL: dict[str, str] = {
    "mối quan hệ": "relational",
    "xác định đối tượng/ thuộc tính": "recognition",
    "xác định thuộc tính/ đối tượng": "recognition",
    "xác định thuộc tính/đối tượng": "recognition",
    "mô tả đối tượng/ thuộc tính": "recognition",
    "mô tả đối tượng/thuộc tính": "recognition",
    "xác định thuộc tính": "recognition",
    "mô tả thuộc tính": "recognition",
    "mô tả vị trí/ không gian": "spatial",
    "lý do/ nhân quả": "causal",
    "mục đích/ chức năng": "causal",
    "mục đích": "causal",
    "mô tả hành động": "action",
    "xác định số lượng": "counting",
    "suy luận ngữ cảnh": "context",
    "câu hỏi có/không": "yesno",
}


def _norm_q(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _parse_answers(value) -> list[str]:
    if isinstance(value, list):
        return [str(a) for a in value]
    try:
        parsed = ast.literal_eval(value)
        return [str(a) for a in parsed] if isinstance(parsed, (list, tuple)) else [str(value)]
    except (ValueError, SyntaxError):
        return [str(value)]


def build(texts_dir: Path, images_dir: Path, out: Path) -> pd.DataFrame:
    labels_path = texts_dir / "final_vqa_dataset.json"
    csv_path = texts_dir / "evaluate_60k_data_balanced_preprocessed.csv"

    labels = pd.DataFrame(json.load(open(labels_path)))
    labels["img_id"] = labels["img_id"].astype(int)
    labels["_q"] = labels["question"].map(_norm_q)
    labels["category"] = labels["category"].map(_norm_label).map(CANONICAL)
    labels = labels.dropna(subset=["category"])
    labels = labels.drop_duplicates(["img_id", "_q"])[["img_id", "_q", "category", "reason_depth"]]

    df = pd.read_csv(csv_path, low_memory=False)
    df["_q"] = df["question"].map(_norm_q)
    merged = df.merge(labels, left_on=["image_id", "_q"], right_on=["img_id", "_q"], how="inner")

    merged["answers"] = merged["answers"].map(_parse_answers)
    merged["image_path"] = merged["image_id"].map(
        lambda i: str(images_dir / f"{int(i):012d}.jpg")
    )
    merged = merged.drop(columns=["_q", "img_id"], errors="ignore")

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    return merged


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data.labeled_table", description=__doc__)
    p.add_argument("--out", default="data/labeled.parquet")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = load_config(repo_root() / "configs" / "data.yaml")["data_path"]
    root = repo_root()
    df = build(root / cfg["raw_texts"], root / cfg["raw_images"], root / args.out)
    print(f"[labeled_table] {len(df)} rows -> {args.out}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
