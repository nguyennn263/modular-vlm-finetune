"""Score the OOD table on the questions both runs share: Vintern-base (original run,
unaffected by the first-tile crop), ours from the original run (first-tile crop) and
ours re-run with --gen-image full. Every metric comes from metrics.vqa_metrics.score_answers,
the same function behind every other table.

    python experiments/ood-eval/score_full.py --old <repo>/outputs/ood_eval --new outputs/ood_full
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from metrics.vqa_metrics import score_answers  # noqa: E402

DATASETS = ["vitextvqa", "openvivqa", "vivqax", "vivqa"]
SEEDS = [42, 123, 3407]
# seed-42 runs of the first three datasets predate multi-seed and have no _s42 suffix
LEGACY = {"vitextvqa", "vivqax", "openvivqa"}


def _key(image: str, question: str) -> tuple[str, str]:
    return image, " ".join(question.lower().split())


def _load(pred_file: Path, rows: list[dict], image_field: str, question_of) -> dict:
    samples = json.loads(pred_file.read_text())["samples"]
    assert len(samples) == len(rows), f"{pred_file}: {len(samples)} preds vs {len(rows)} rows"
    return {_key(r[image_field], question_of(r)): s for r, s in zip(rows, samples)}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def _score(samples: list[dict]) -> dict:
    avg, _ = score_answers([s["prediction"] for s in samples], [s["ground_truths"] for s in samples])
    return {k: 100 * avg[k] for k in ("accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    table = {}
    for ds in DATASETS:
        per_seed = []
        for seed in SEEDS:
            old_tag = ds if (seed == 42 and ds in LEGACY) else f"{ds}_s{seed}"
            new_tag = f"{ds}_s{seed}"
            old_dir, new_dir = a.old / old_tag, next((a.new / new_tag).rglob("ours.jsonl"), None)
            if new_dir is None:
                print(f"[skip] {new_tag}: not collected yet"); continue
            new_dir = new_dir.parent
            old_ours_rows = _jsonl(old_dir / "data" / old_tag / "ours.jsonl")
            vintern = _load(old_dir / "out" / old_tag / "vintern_base" / "results" / "text_predictions_epoch_1.json",
                            _jsonl(old_dir / "data" / old_tag / "internvl.jsonl"), "image",
                            lambda r: r["conversations"][0]["value"].split("\n", 1)[-1])
            ours_crop = _load(old_dir / "out" / old_tag / "ours" / "results" / "text_predictions_epoch_1.json",
                              old_ours_rows, "image_name", lambda r: r["question"])
            ours_full = _load(new_dir / "text_predictions_epoch_1.json", _jsonl(new_dir / "ours.jsonl"),
                              "image_name", lambda r: r["question"])
            shared = sorted(set(vintern) & set(ours_crop) & set(ours_full))
            row = {"seed": seed, "n": len(shared),
                   "vintern": _score([vintern[k] for k in shared]),
                   "ours_crop": _score([ours_crop[k] for k in shared]),
                   "ours_full": _score([ours_full[k] for k in shared])}
            per_seed.append(row)
            print(f"{ds:10s} s{seed:<5d} n={row['n']:4d}  F1 vintern {row['vintern']['f1']:6.2f}  "
                  f"ours crop {row['ours_crop']['f1']:6.2f}  ours full {row['ours_full']['f1']:6.2f}  |  "
                  f"CIDEr {row['vintern']['cider']:6.1f} / {row['ours_crop']['cider']:6.1f} / "
                  f"{row['ours_full']['cider']:6.1f}")
        if per_seed:
            agg = {}
            for model in ("vintern", "ours_crop", "ours_full"):
                for metric in ("accuracy", "precision", "recall", "f1", "bleu", "rouge_l", "meteor", "cider"):
                    vals = [r[model][metric] for r in per_seed]
                    agg[f"{model}.{metric}"] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
            table[ds] = {"seeds": per_seed, "mean_std": agg}
    if a.out:
        a.out.write_text(json.dumps(table, indent=2, ensure_ascii=False))
        print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
