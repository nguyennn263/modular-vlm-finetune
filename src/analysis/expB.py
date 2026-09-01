"""Exp B — bridge x reasoning-type, and the fork decision (final-plan P3).

    python -m src.analysis.expB --glob "checkpoints/expA/**/eval_val_samples.jsonl"

Inputs: the per-sample JSONL files written by ``src.cli.evaluate`` (one per
bridge, seeds pooled). For each ``category`` it ranks bridges by mean CIDEr and
runs a **paired bootstrap** (best vs 2nd best) on the samples both saw — this
p-value (Holm-adjusted across categories) is the ONLY decision signal. Kendall's
W and "how often the top bridge changes" are descriptive only.

Outputs ``outputs/expB/{summary.json, heatmap.csv}`` and picks the top-3 bridges
(CIDEr weighted by the TRAIN category distribution) for ``configs/action_space.yaml``.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.analysis.stats import holm, paired_bootstrap
from src.config.loader import repo_root
from src.reasoning_types import CATEGORIES

METRIC = "cider"


def _load(glob_pat: str) -> list[dict]:
    rows = []
    for f in glob.glob(str(repo_root() / glob_pat), recursive=True):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No per-sample files matched {glob_pat} — run Exp A + evaluate first.")
    return rows


def _train_category_weights(split_dir: str) -> dict[str, float]:
    from src.data.split import load_split
    c = Counter(s.metadata["category"] for s in load_split("train", split_dir))
    total = sum(c.values())
    return {k: c.get(k, 0) / total for k in CATEGORIES}


def analyse(rows: list[dict], weights: dict[str, float]) -> dict:
    # (bridge, category) -> {(image_id, question): score}
    cell: dict = defaultdict(dict)
    for r in rows:
        if r.get(METRIC) is None or r.get("category") not in CATEGORIES:
            continue
        cell[(r["bridge"], r["category"])][(r["image_id"], r["question"])] = r[METRIC]

    bridges = sorted({b for b, _ in cell})
    heatmap = {c: {b: float(np.mean(list(cell[(b, c)].values()))) if cell[(b, c)] else None
                   for b in bridges} for c in CATEGORIES}

    fork = {}
    raw_p = {}
    for c in CATEGORIES:
        ranked = sorted((b for b in bridges if cell[(b, c)]),
                        key=lambda b: heatmap[c][b], reverse=True)
        if len(ranked) < 2:
            continue
        top, second = ranked[0], ranked[1]
        common = sorted(set(cell[(top, c)]) & set(cell[(second, c)]))
        if len(common) < 20:
            fork[c] = {"top": top, "second": second, "note": "too few common samples"}
            continue
        a = np.array([cell[(top, c)][k] for k in common])
        b = np.array([cell[(second, c)][k] for k in common])
        bs = paired_bootstrap(a, b)
        raw_p[c] = bs["p_value"]
        fork[c] = {"top": top, "second": second, "rank": ranked,
                   "mean_cider": {k: round(heatmap[c][k], 2) for k in ranked},
                   **{k: round(v, 4) for k, v in bs.items() if isinstance(v, float)}}

    adj = holm(raw_p)
    for c, p in adj.items():
        fork[c]["p_holm"] = round(p, 4)
        fork[c]["significant"] = bool(p < 0.05)

    # descriptive: how often does the best bridge change across categories?
    tops = [fork[c]["top"] for c in fork if "top" in fork[c]]
    top_changes = len(set(tops))

    weighted = {b: sum(weights[c] * (heatmap[c][b] or 0.0) for c in CATEGORIES) for b in bridges}
    top3 = sorted(bridges, key=lambda b: weighted[b], reverse=True)[:3]

    return {
        "metric": METRIC,
        "bridges": bridges,
        "heatmap": heatmap,
        "fork": fork,
        "n_categories_significant": sum(v.get("significant", False) for v in fork.values()),
        "distinct_top_bridges": top_changes,
        "weighted_cider": {b: round(v, 3) for b, v in weighted.items()},
        "top3_for_action_space": top3,
        "verdict": (
            "bridge choice matters per reasoning-type (fork)"
            if sum(v.get("significant", False) for v in fork.values()) >= 2
            else "no strong per-type fork — use the single best bridge + focus on the n_tiles lever"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.analysis.expB", description=__doc__)
    p.add_argument("--glob", default="checkpoints/expA/**/eval_val_samples.jsonl")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--out", default="outputs/expB")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    rows = _load(args.glob)
    result = analyse(rows, _train_category_weights(args.split_dir))

    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    with open(out / "heatmap.csv", "w", encoding="utf-8") as fh:
        fh.write("category," + ",".join(result["bridges"]) + "\n")
        for c in CATEGORIES:
            fh.write(c + "," + ",".join(
                f"{result['heatmap'][c][b]:.2f}" if result["heatmap"][c][b] is not None else ""
                for b in result["bridges"]) + "\n")

    print(json.dumps({
        "verdict": result["verdict"],
        "significant_categories": result["n_categories_significant"],
        "top3_for_action_space": result["top3_for_action_space"],
        "weighted_cider": result["weighted_cider"],
    }, indent=2, ensure_ascii=False))
    print(f"[expB] {out}/summary.json + heatmap.csv")


if __name__ == "__main__":
    main()
