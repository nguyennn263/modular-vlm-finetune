"""C4 — self-check sample for the "human validation" pending item.

Originally scoped as 300-500 samples / 2 annotators / Cohen's kappa (needs
the user's time, not available before the deadline). User authorized doing
this as a single-rater spot-check instead ("m tu check luon di"). This script
does the sampling half: stratified by category x F1-bucket, on the PLAIN
multi_token headline bridge (the paper's main claim, not the LoRA reference
point) so the check validates what the paper actually leads with.

    python scripts/human_validation_sample.py --n-per-cat 15
    -> outputs/human_validation/sample_for_review.json

The judging half (reading each sample, assigning correct/partial/incorrect/
nonsensical) is done by the assistant in the same session, not by this
script -- there is no automated "judge" here on purpose.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.config.loader import repo_root

_PUNC = str.maketrans("", "", string.punctuation)


def _norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", str(s)).translate(_PUNC).lower()
    return s.split()


def _f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pred), n / len(ref)
    return 2 * p * r / (p + r)


def _bucket(f1: float) -> str:
    if f1 >= 0.6:
        return "strong"
    if f1 >= 0.2:
        return "partial"
    if f1 > 0:
        return "weak"
    return "zero"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="multi_token")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-cat", type=int, default=15)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    root = repo_root()
    base = root / f"checkpoints/expA/seed{args.seed}/{args.bridge}"
    preds = json.loads((base / "results/text_predictions_epoch_1.json").read_text())["samples"]
    evs = [json.loads(x) for x in (base / "eval_val_samples.jsonl").read_text().splitlines() if x.strip()]
    cats = [e.get("category", "?") for e in evs] if len(evs) == len(preds) else None
    if cats is None:
        by_q = defaultdict(list)
        for e in evs:
            by_q[e["question"]].append(e.get("category", "?"))
        cats = [by_q.get(s["question"], ["?"])[0] for s in preds]

    rows = []
    for i, s in enumerate(preds):
        p = _norm(s["prediction"])
        refs = [_norm(r) for r in s["ground_truths"]]
        f1 = max((_f1(p, r) for r in refs), default=0.0)
        rows.append({
            "idx": i, "category": cats[i], "question": s["question"],
            "prediction": s["prediction"], "ground_truths": s["ground_truths"],
            "f1": round(f1, 3), "bucket": _bucket(f1),
        })

    by_cat_bucket: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat_bucket[r["category"]][r["bucket"]].append(r)

    rng = random.Random(args.rng_seed)
    sample = []
    for cat, buckets in sorted(by_cat_bucket.items()):
        # proportional-to-population draw across buckets within this category,
        # so the sample mirrors the real F1 distribution instead of forcing
        # an artificial even split (which would bias "how good is it really")
        pool = [r for bucket_rows in buckets.values() for r in bucket_rows]
        rng.shuffle(pool)
        sample.extend(pool[: args.n_per_cat])

    rng.shuffle(sample)  # de-correlate review order from category, avoid rater drift/anchoring
    out = {
        "bridge": args.bridge, "seed": args.seed, "n_total_pool": len(rows),
        "n_sampled": len(sample), "n_per_cat_requested": args.n_per_cat,
        "note": "Single-rater (assistant) spot-check sample. NOT the originally-scoped "
                "300-500/2-annotator/Cohen's-kappa protocol -- report accordingly.",
        "samples": sample,
    }
    dst = root / "outputs/human_validation/sample_for_review.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"sampled {len(sample)} / {len(rows)} across {len(by_cat_bucket)} categories -> {dst}")
    for cat, buckets in sorted(by_cat_bucket.items()):
        print(f"  {cat:14} n={sum(len(v) for v in buckets.values()):5} "
              f"buckets={{{', '.join(f'{k}:{len(v)}' for k, v in buckets.items())}}}")


if __name__ == "__main__":
    main()
