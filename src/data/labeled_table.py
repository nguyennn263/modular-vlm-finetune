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


def _load_labels_json() -> list:
    """Reasoning-type labels. Bundled gzipped in the repo (assets/), so this works
    identically locally and on a fresh Kaggle clone."""
    import gzip

    for cand in [
        repo_root() / "assets" / "final_vqa_dataset.json.gz",
        repo_root() / "data" / "raw" / "texts" / "final_vqa_dataset.json",
        *(Path("/kaggle/input").glob("*/final_vqa_dataset.json*") if Path("/kaggle/input").exists() else []),
    ]:
        if cand.exists():
            opener = gzip.open if cand.suffix == ".gz" else open
            with opener(cand, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    raise SystemExit("final_vqa_dataset.json(.gz) not found (assets/ or data/raw/texts/).")


def build(texts_dir: Path, images_dir: Path, out: Path) -> pd.DataFrame:
    csv_path = texts_dir / "evaluate_60k_data_balanced_preprocessed.csv"

    labels = pd.DataFrame(_load_labels_json())
    labels["img_id"] = labels["img_id"].astype(int)
    labels["_q"] = labels["question"].map(_norm_q)
    labels["category"] = labels["category"].map(_norm_label).map(CANONICAL)
    labels = labels.dropna(subset=["category"])
    labels = labels.drop_duplicates(["img_id", "_q"])[["img_id", "_q", "category", "reason_depth"]]

    df = pd.read_csv(csv_path, low_memory=False)
    df["_q"] = df["question"].map(_norm_q)
    merged = df.merge(labels, left_on=["image_id", "_q"], right_on=["img_id", "_q"], how="inner")

    merged["answers"] = merged["answers"].map(_parse_answers)
    # Store the basename only; the image dir is resolved per-environment at load time.
    merged["image_name"] = merged["image_id"].map(lambda i: f"{int(i):012d}.jpg")
    merged = merged.drop(columns=["_q", "img_id"], errors="ignore")

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    return merged


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data.labeled_table", description=__doc__)
    p.add_argument("--out", default="data/labeled.parquet")
    return p


def resolve_dirs() -> tuple[Path, Path]:
    """(texts_dir, images_dir), environment-aware (local or Kaggle mount)."""
    from src.data.environment import DataPathResolver

    cfg = load_config(repo_root() / "configs" / "data.yaml")
    r = DataPathResolver(cfg, cfg["kaggle_setup"], str(repo_root()))
    return r.get_raw_texts_file().parent, r.get_raw_images_dir()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    texts_dir, images_dir = resolve_dirs()
    df = build(texts_dir, images_dir, repo_root() / args.out)
    print(f"[labeled_table] {len(df)} rows -> {args.out}  (texts={texts_dir})")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
