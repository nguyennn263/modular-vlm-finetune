#!/usr/bin/env python3
"""P0 - environment & data: labelled table + 70/15/15 grouped split."""
import sys

from _phase import Step, run_phase

PY = sys.executable

STEPS = [
    Step("Download AutoViVQA into data/raw/ (skip on Kaggle)",
         [PY, "-m", "src.cli.download"]),
    Step("Build labelled table (join final_vqa_dataset.json + CSV)",
         [PY, "-m", "src.data.labeled_table", "--out", "data/labeled.parquet"]),
    Step("Build 70/15/15 split grouped by image, stratified by category",
         [PY, "-m", "src.data.split", "--ratios", "0.7", "0.15", "0.15", "--seed", "42"]),
]

if __name__ == "__main__":
    run_phase("phase0_build_data", STEPS)
