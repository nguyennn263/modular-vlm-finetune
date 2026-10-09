"""Score the 2-annotator human-validation study -> outputs/human_validation/report_kappa.md

    python scripts/human_validation_kappa.py

Reads (all under outputs/human_validation/):
    annotation_form_A.csv, annotation_form_B.csv   filled by the two annotators
    answer_key.json                                hidden per-item token-F1 + category

Reports:
    - per-rater label distribution
    - Cohen's kappa, 4-way and collapsed to acceptable (correct+partial) vs not
    - consensus label distribution (items both raters agree on) + adjudication gap
    - token-F1 bucket x consensus judgment cross-tab  (re-checks the self-check's
      "partial-overlap band is only ~43% acceptable" claim, now with images + 2 raters)
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HV = ROOT / "outputs" / "human_validation"
LABELS = ["correct", "partial", "incorrect", "nonsensical"]
JCOL = "judgment (correct | partial | incorrect | nonsensical)"


def _read_form(name: str) -> dict[int, str]:
    out = {}
    with open(HV / name, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            j = (row.get(JCOL) or "").strip().lower()
            if j:
                out[int(row["id"])] = j
    return out


def _cohen_kappa(pairs: list[tuple[str, str]], cats: list[str]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def _bucket(f1: float) -> str:
    return ("zero" if f1 == 0 else "weak" if f1 < 0.2 else
            "partial" if f1 <= 0.6 else "strong")


def main() -> None:
    key = json.loads((HV / "answer_key.json").read_text())
    f1_by_id = {it["id"]: it["token_f1"] for it in key["items"]}
    cat_by_id = {it["id"]: it["category"] for it in key["items"]}

    a, b = _read_form("annotation_form_A.csv"), _read_form("annotation_form_B.csv")
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("no rows scored in BOTH forms yet -- fill annotation_form_{A,B}.csv first")
    bad = ({v for v in a.values()} | {v for v in b.values()}) - set(LABELS)
    if bad:
        print(f"WARNING: unrecognised labels ignored in kappa cats: {bad}")

    pairs = [(a[i], b[i]) for i in ids]
    L = [x for x in LABELS]
    k4 = _cohen_kappa(pairs, L)

    def acc(x):  # collapse to binary
        return "acceptable" if x in ("correct", "partial") else "not"
    k2 = _cohen_kappa([(acc(x), acc(y)) for x, y in pairs], ["acceptable", "not"])

    agree = [i for i in ids if a[i] == b[i]]
    consensus = {i: a[i] for i in agree}

    out = [
        "# Human validation -- 2-annotator study",
        "",
        f"**Model:** {key['model']}  ",
        f"**Predictions:** `{key['pred_file']}`  ",
        f"**Items scored by both raters:** {len(ids)} / {key['n']}",
        "",
        "## Per-rater label distribution",
        "",
        "| label | rater A | rater B |",
        "|---|---:|---:|",
    ]
    ca, cb = Counter(a[i] for i in ids), Counter(b[i] for i in ids)
    for lab in LABELS:
        out.append(f"| {lab} | {ca[lab]} ({100*ca[lab]/len(ids):.1f}%) | {cb[lab]} ({100*cb[lab]/len(ids):.1f}%) |")

    out += [
        "",
        "## Inter-rater agreement",
        "",
        f"- raw agreement (4-way): **{100*len(agree)/len(ids):.1f}%** ({len(agree)}/{len(ids)})",
        f"- Cohen's kappa (4-way): **{k4:.3f}**",
        f"- Cohen's kappa (acceptable vs not): **{k2:.3f}**",
        "",
        "## Consensus judgment (items both raters agree on)",
        "",
        "| label | n | % of consensus |",
        "|---|---:|---:|",
    ]
    cc = Counter(consensus.values())
    for lab in LABELS:
        out.append(f"| {lab} | {cc[lab]} | {100*cc[lab]/len(agree):.1f}% |")
    acc_n = cc["correct"] + cc["partial"]
    out.append(f"| **acceptable (correct+partial)** | **{acc_n}** | **{100*acc_n/len(agree):.1f}%** |")

    # token-F1 bucket x consensus judgment
    out += [
        "",
        "## token-F1 bucket x consensus judgment",
        "",
        "Buckets: zero (F1=0), weak (<0.2), partial (0.2-0.6), strong (>0.6). "
        "Re-checks the self-check finding that the partial-overlap band is mostly "
        "not semantically acceptable.",
        "",
        "| F1 bucket | n | correct | partial | incorrect | nonsensical | acceptable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_b = defaultdict(list)
    for i in agree:
        by_b[_bucket(f1_by_id[i])].append(consensus[i])
    for bk in ["strong", "partial", "weak", "zero"]:
        js = by_b.get(bk, [])
        if not js:
            continue
        c = {l: js.count(l) for l in LABELS}
        ok = c["correct"] + c["partial"]
        out.append(f"| {bk} | {len(js)} | {c['correct']} | {c['partial']} | {c['incorrect']} "
                   f"| {c['nonsensical']} | {ok} ({100*ok/len(js):.1f}%) |")

    # per-category acceptable rate (consensus)
    out += ["", "## Consensus acceptable-rate by reasoning category", "",
            "| category | n | acceptable |", "|---|---:|---:|"]
    by_c = defaultdict(list)
    for i in agree:
        by_c[cat_by_id[i]].append(consensus[i])
    for cat, js in sorted(by_c.items(), key=lambda kv: -len(kv[1])):
        ok = sum(1 for x in js if x in ("correct", "partial"))
        out.append(f"| {cat} | {len(js)} | {ok} ({100*ok/len(js):.1f}%) |")

    (HV / "report_kappa.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {HV.relative_to(ROOT)}/report_kappa.md")


if __name__ == "__main__":
    main()
