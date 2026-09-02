"""Re-score saved Exp A predictions with corpus-level standard CIDEr-D
(metrics/cider/cider.py) + the other metrics/ scorers, offline (no GPU).

The training-time `metrics.vqa_metrics.CIDErScore` is a simplified per-batch
metric (normalised cosine, no length penalty, ~0.1 range x10). This script
recomputes CIDEr the pycocoevalcap way over the whole split at once so the
numbers are comparable to plans/results-5bridge.md and other papers.

    python scripts/rescore_expA.py --seed 42
    -> outputs/expA/rescored_seed42.{json,md}
"""
from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

from src.config.loader import repo_root
from metrics.cider.cider import Cider


def _norm(s: str) -> str:
    s = str(s).translate(str.maketrans("", "", string.punctuation)).lower().strip()
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.split())


def corpus_cider(preds: list[str], refs: list[list[str]]) -> tuple[float, list[float]]:
    """Standard CIDEr-D: one shared corpus (IDF over all samples), mean over refs."""
    gts = {str(i): [_norm(r) for r in rs] for i, rs in enumerate(refs)}
    res = {str(i): [_norm(p)] for i, p in enumerate(preds)}
    score, per = Cider().compute_score(gts, res)
    return float(score), [float(x) for x in per]


def _load_preds(bridge_dir: Path) -> list[dict]:
    f = bridge_dir / "results" / "text_predictions_epoch_1.json"
    if not f.exists():
        cands = sorted(bridge_dir.glob("results/text_predictions_epoch_*.json"))
        f = cands[-1] if cands else None
    if f is None:
        raise SystemExit(f"no text_predictions_epoch_*.json under {bridge_dir}")
    d = json.loads(f.read_text())
    return d["samples"] if isinstance(d, dict) and "samples" in d else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-dir", default="checkpoints/expA")
    ap.add_argument("--split-dir", default="data/splits")
    a = ap.parse_args()

    root = repo_root()
    base = root / a.ckpt_dir / f"seed{a.seed}"
    bridges = sorted(p.name for p in base.iterdir() if (p / "results").is_dir())
    if not bridges:
        raise SystemExit(f"no bridges with results/ under {base}")

    # category per question text from the val split (text_predictions rows carry
    # no image_id, but questions are ~unique and always carry a category)
    cat = {}
    for line in (root / a.split_dir / "val.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            meta = r.get("metadata", r)
            cat[r["question"]] = meta.get("category")

    out = {"seed": a.seed, "metric": "corpus CIDEr-D (metrics/cider)", "bridges": {}}
    for b in bridges:
        rows = _load_preds(base / b)
        preds = [r["prediction"] for r in rows]
        refs = [r["ground_truths"] for r in rows]
        overall, per = corpus_cider(preds, refs)

        by_cat: dict[str, list[float]] = {}
        for r, s in zip(rows, per):
            c = cat.get(r.get("question"), "?")
            by_cat.setdefault(c, []).append(s)
        out["bridges"][b] = {
            "cider_corpus": round(overall, 2),
            "n": len(rows),
            "by_category": {c: round(sum(v) / len(v), 2) for c, v in sorted(by_cat.items())},
        }
        print(f"{b:16} CIDEr-D {overall:6.2f}  (n={len(rows)})")

    dst = root / "outputs" / "expA" / f"rescored_seed{a.seed}.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # markdown table
    cats = ["relational", "counting", "spatial", "recognition",
            "action", "causal", "context", "yesno"]
    lines = [f"# Exp A seed {a.seed} — corpus CIDEr-D re-score\n",
             "| bridge | overall | " + " | ".join(cats) + " |",
             "|" + "---|" * (len(cats) + 2)]
    for b, v in sorted(out["bridges"].items(), key=lambda kv: -kv[1]["cider_corpus"]):
        row = [b, f"**{v['cider_corpus']}**"] + [f"{v['by_category'].get(c, '-')}" for c in cats]
        lines.append("| " + " | ".join(row) + " |")
    (root / "outputs" / "expA" / f"rescored_seed{a.seed}.md").write_text("\n".join(lines) + "\n")
    print(f"\n-> {dst}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
