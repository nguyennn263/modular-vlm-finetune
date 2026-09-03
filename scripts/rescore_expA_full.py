"""Full baseline-comparable rescore of Exp A predictions.

Runs metrics/compute_score.compute_all_data (the SAME function that produced
plans/results-5bridge.md and the ViMoE baseline table) on the saved full-val
epoch-1 predictions for every bridge, so the numbers are directly comparable
to the baseline table (per-sample max-over-refs, then averaged).

    python scripts/rescore_expA_full.py --seed 42
    -> outputs/expA/rescored_full_seed42.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.loader import repo_root
from metrics.compute_score import compute_all_data

BRIDGES = ["multi_token", "qformer", "mini_qformer", "tile_attention", "residual"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--bridges", default=",".join(BRIDGES))
    args = ap.parse_args()

    root = repo_root()
    out: dict = {}
    for b in [x.strip() for x in args.bridges.split(",")]:
        pf = root / f"checkpoints/expA/seed{args.seed}/{b}/results/text_predictions_epoch_{args.epoch}.json"
        if not pf.exists():
            print(f"[skip] {b}: {pf} missing")
            continue
        samples = json.loads(pf.read_text())["samples"]
        gts = [s["ground_truths"] for s in samples]
        gen = [s["prediction"] for s in samples]
        print(f"[{b}] {len(samples)} samples -> compute_all_data ...")
        sc = compute_all_data(gts, gen)
        out[b] = {k: float(v["average"] if isinstance(v, dict) else v) for k, v in sc.items()}
        out[b]["n"] = len(samples)
        print(f"[{b}] {out[b]}")
        (root / f"outputs/expA/rescored_full_seed{args.seed}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))

    # markdown table (x100)
    cols = ["accuracy", "precision", "recall", "f1_token", "bleu", "rouge", "meteor", "cider"]
    lines = ["# Exp A full rescore (metrics/compute_score.compute_all_data, x100)", "",
             "| bridge | Acc | Prec | Rec | F1 | BLEU | ROUGE | METEOR | CIDEr |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b, sc in out.items():
        lines.append("| " + b + " | " + " | ".join(f"{sc[c]*100:.2f}" for c in cols) + " |")
    (root / f"outputs/expA/rescored_full_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
