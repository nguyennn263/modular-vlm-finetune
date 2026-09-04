"""C5 — quantitative error analysis of a bridge's val predictions.

    python scripts/error_analysis.py --bridge multi_token

Joins checkpoints/expA/seed42/<bridge>/results/text_predictions_epoch_1.json
(prediction + 5 refs, full val) with the grouped-split category labels, then:
  - length: predicted vs reference token counts (the concise-generation effect)
  - per-category token-F1 (max over 5 refs)
  - error buckets: near-miss / partial / total-miss by token-F1 thresholds
  - counting-question noun-omission rate (ViMoE reports 10.7%)
  - a handful of qualitative examples per bucket
Writes outputs/expA/error_analysis_<bridge>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.config.loader import repo_root
from src.data.split import load_split

_PUNC = str.maketrans("", "", string.punctuation)
_NUMWORDS = {
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "không", "vài", "nhiều", "mấy",
}


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


def _has_number(toks: list[str]) -> bool:
    return any(t.isdigit() or t in _NUMWORDS for t in toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="multi_token")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = repo_root()

    preds = json.loads(
        (root / f"checkpoints/expA/seed{args.seed}/{args.bridge}/results/text_predictions_epoch_1.json").read_text()
    )["samples"]
    # eval_val_samples.jsonl is 1:1 and in the same order as text_predictions,
    # and carries `category` directly. Fall back to a (image_id, question) join.
    evs_path = root / f"outputs/expA/seed{args.seed}/{args.bridge}/eval_val_samples.jsonl"
    if not evs_path.exists():
        evs_path = root / f"checkpoints/expA/seed{args.seed}/{args.bridge}/eval_val_samples.jsonl"
    evs = [json.loads(x) for x in evs_path.read_text().splitlines() if x.strip()]
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
        best = max((_f1(p, r) for r in refs), default=0.0)
        ref_len = sum(len(r) for r in refs) / max(1, len(refs))
        cat = cats[i]
        # noun omission: counting Q, pred has a number, but shares no non-number
        # content token with any ref
        ref_nouns = {t for r in refs for t in r if not (t.isdigit() or t in _NUMWORDS)}
        noun_omit = (
            cat == "counting"
            and _has_number(p)
            and not ({t for t in p if not (t.isdigit() or t in _NUMWORDS)} & ref_nouns)
        )
        rows.append({
            "q": s["question"], "pred": s["prediction"], "refs": s["ground_truths"],
            "cat": cat, "f1": best, "pred_len": len(p), "ref_len": ref_len,
            "noun_omit": noun_omit,
        })

    n = len(rows)
    out = {"bridge": args.bridge, "n": n}

    # length
    out["mean_pred_len"] = round(sum(r["pred_len"] for r in rows) / n, 2)
    out["mean_ref_len"] = round(sum(r["ref_len"] for r in rows) / n, 2)
    out["pred_shorter_than_ref_pct"] = round(
        100 * sum(r["pred_len"] < r["ref_len"] for r in rows) / n, 1)

    # buckets
    buckets = {"strong (F1>=.6)": 0, "partial (.2-.6)": 0, "weak (0-.2)": 0, "zero (F1=0)": 0}
    for r in rows:
        f = r["f1"]
        if f >= 0.6:
            buckets["strong (F1>=.6)"] += 1
        elif f >= 0.2:
            buckets["partial (.2-.6)"] += 1
        elif f > 0:
            buckets["weak (0-.2)"] += 1
        else:
            buckets["zero (F1=0)"] += 1
    out["buckets_pct"] = {k: round(100 * v / n, 1) for k, v in buckets.items()}

    # per-category
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r["f1"])
    out["per_category_f1"] = {
        c: {"n": len(v), "mean_f1": round(sum(v) / len(v), 3)}
        for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
    }

    # counting noun omission
    cnt = [r for r in rows if r["cat"] == "counting"]
    n_omit = sum(r["noun_omit"] for r in cnt)
    out["counting_noun_omission"] = {
        "n_counting": len(cnt),
        "n_omit": n_omit,
        "pct": round(100 * n_omit / max(1, len(cnt)), 1),
    }

    # qualitative samples
    def _samp(pred_key, k=6):
        return [
            {"q": r["q"], "pred": r["pred"], "refs": r["refs"], "cat": r["cat"], "f1": round(r["f1"], 2)}
            for r in rows if pred_key(r)
        ][:k]

    out["examples"] = {
        "zero_f1": _samp(lambda r: r["f1"] == 0),
        "partial_f1": _samp(lambda r: 0.2 <= r["f1"] < 0.5),
        "noun_omission": _samp(lambda r: r["noun_omit"]),
    }

    (root / f"outputs/expA/error_analysis_{args.bridge}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    # markdown
    L = [f"# Error analysis — {args.bridge} (val, seed {args.seed}, {n} samples)", ""]
    L.append(f"- Mean predicted length **{out['mean_pred_len']}** tokens vs reference **{out['mean_ref_len']}**; "
             f"prediction shorter than the mean reference in **{out['pred_shorter_than_ref_pct']}%** of cases.")
    L.append("")
    L.append("## Token-F1 buckets (max over 5 refs)")
    L.append("| bucket | % |")
    L.append("|---|---:|")
    for k, v in out["buckets_pct"].items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Per reasoning-type")
    L.append("| category | n | mean token-F1 |")
    L.append("|---|---:|---:|")
    for c, d in out["per_category_f1"].items():
        L.append(f"| {c} | {d['n']} | {d['mean_f1']} |")
    L.append("")
    co = out["counting_noun_omission"]
    L.append(f"## Counting questions: noun omission = **{co['pct']}%** ({co['n_omit']}/{co['n_counting']}) "
             f"— cf. ViMoE-VQA 10.7%.")
    L.append("")
    L.append("## Examples — zero token-F1")
    for e in out["examples"]["zero_f1"]:
        L.append(f"- *[{e['cat']}]* Q: {e['q']}")
        L.append(f"  - pred: **{e['pred']}**  | refs: {e['refs']}")
    (root / f"outputs/expA/error_analysis_{args.bridge}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
