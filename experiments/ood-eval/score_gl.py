"""OOD table with the global-local bridge: Vintern-1B zero-shot (6 tiles, original run),
Multi-Token + LoRA 3 ep (whole image, ood_full) and ViBridge-VQA g14-k36 / g14-k144
(ood_gl), scored on the questions all four share, with metrics.vqa_metrics.score_answers.

    python experiments/ood-eval/score_gl.py --old <main checkout>/outputs/ood_eval \
        --full outputs/ood_full --gl outputs/ood_gl --out outputs/ood_gl/table.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_full import DATASETS, LEGACY, SEEDS, _jsonl, _load, _score  # noqa: E402

MODELS = ("vintern", "lora3ep", "gl_k36", "gl_k144")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--gl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    table = {}
    for ds in DATASETS:
        per_seed = []
        for seed in SEEDS:
            tag = f"{ds}_s{seed}"
            old_tag = ds if (seed == 42 and ds in LEGACY) else tag
            old_dir = a.old / old_tag
            full = a.full / tag / "out" / tag / "ours.jsonl"
            full = full if full.exists() else None
            gl = next((a.gl / ds).rglob(f"{tag}/ours.jsonl"), None)
            if full is None or gl is None:
                print(f"[skip] {tag}: full={full} gl={gl}")
                continue
            q = lambda r: r["question"]
            preds = {
                "vintern": _load(old_dir / "out" / old_tag / "vintern_base" / "results" / "text_predictions_epoch_1.json",
                                 _jsonl(old_dir / "data" / old_tag / "internvl.jsonl"), "image",
                                 lambda r: r["conversations"][0]["value"].split("\n", 1)[-1]),
                "lora3ep": _load(full.parent / "text_predictions_epoch_1.json", _jsonl(full), "image_name", q),
            }
            for m in ("k36", "k144"):
                preds[f"gl_{m}"] = _load(gl.parent / m / "text_predictions_epoch_1.json", _jsonl(gl), "image_name", q)
            shared = sorted(set.intersection(*(set(p) for p in preds.values())))
            row = {"seed": seed, "n": len(shared), **{m: _score([preds[m][k] for k in shared]) for m in MODELS}}
            per_seed.append(row)
            print(f"{ds:10s} s{seed:<5d} n={row['n']:4d}  F1 " +
                  "  ".join(f"{m} {row[m]['f1']:6.2f}" for m in MODELS))
        if per_seed:
            table[ds] = {"seeds": per_seed, "mean_std": {
                # population std (ddof=0), as every table of the paper (incl. its earlier OOD table)
                f"{m}.{metric}": (st.mean(r[m][metric] for r in per_seed),
                                  st.pstdev([r[m][metric] for r in per_seed]))
                for m in MODELS for metric in per_seed[0][MODELS[0]]}}
    a.out.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
