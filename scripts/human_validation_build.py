#!/usr/bin/env python3
"""Build the real human-validation annotation forms (300 samples, 2 raters, Cohen's kappa).

Upgrades the single-rater N=120 self-check (scripts/human_validation_sample.py,
outputs/human_validation/selfcheck_judgments.json) to a proper study on the SAME
model (plain multi_token bridge, seed 42) so §7 goes cleanly from 1 rater / no
image to 2 independent raters / with image + Cohen's kappa.

- 300 val questions, stratified proportionally by reasoning-type category
- each item shows the COCO image URL, the question, the model's answer, and all
  five references; the annotator assigns a 4-level judgment + optional note
- two identical blank forms (A, B) for independent annotation; answer_key.json
  keeps the per-sample token-F1 hidden from raters so we can report kappa AND
  the F1-vs-human-judgment correlation afterwards (the §7 "partial bucket" claim)

    python scripts/human_validation_build.py
    python scripts/human_validation_build.py --n 300 --pred <path> --model-label "..."

Outputs -> outputs/human_validation/:
    annotation_form_A.csv, annotation_form_B.csv   (give to the two annotators)
    answer_key.json                                (do NOT show annotators)
    ANNOTATION_README.md                           (rater instructions + rubric)
Scoring after both forms are filled: scripts/human_validation_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HV = ROOT / "outputs" / "human_validation"
_PUNC = str.maketrans("", "", string.punctuation)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s).strip())


def _norm(s: str) -> list[str]:
    return unicodedata.normalize("NFC", str(s)).translate(_PUNC).lower().split()


def _f1(pred: str, refs: list[str]) -> float:
    p = _norm(pred)
    best = 0.0
    for r in refs:
        g = _norm(r)
        if not p or not g:
            continue
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            prec, rec = common / len(p), common / len(g)
            best = max(best, 2 * prec * rec / (prec + rec))
    return round(best, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred", default=str(
        ROOT / "checkpoints/expA/seed42/multi_token/results/text_predictions_epoch_1.json"))
    ap.add_argument("--model-label", default="multi_token bridge, frozen backbone, seed 42 (plain, 2 epochs)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    HV.mkdir(parents=True, exist_ok=True)

    pred_raw = json.loads(Path(a.pred).read_text())
    preds = pred_raw.get("samples", pred_raw)
    assert isinstance(preds, list) and preds and "prediction" in preds[0], "unexpected pred format"
    print(f"loaded {len(preds)} predictions from {Path(a.pred).relative_to(ROOT)}")

    # question -> (COCO url, reasoning-type category) from the source dataset
    url_by_q, cat_by_q = {}, {}
    for row in json.loads((ROOT / "data/raw/texts/final_vqa_dataset.json").read_text()):
        q = _nfc(row["question"])
        url_by_q.setdefault(q, row.get("url"))
        cat_by_q.setdefault(q, row.get("category"))

    items, missing = [], 0
    for pr in preds:
        q = _nfc(pr["question"])
        if q not in url_by_q:
            missing += 1
            continue
        refs = pr.get("ground_truths") or pr.get("references") or []
        items.append({
            "image_url": url_by_q[q], "category": cat_by_q.get(q) or "unknown",
            "question": pr["question"], "model_answer": pr["prediction"], "refs": refs,
            "f1": _f1(pr["prediction"], refs),
        })
    if missing:
        print(f"WARNING: {missing} predictions had no dataset match (skipped)")

    # proportional stratified sample by category
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    total = len(items)
    chosen = []
    for cat, group in by_cat.items():
        k = max(2, round(a.n * len(group) / total))
        rng.shuffle(group)
        chosen += group[:k]
    rng.shuffle(chosen)
    chosen = chosen[:a.n]

    # write two identical blank forms + a hidden key
    fields = ["id", "image_url", "question", "model_answer",
              "ref_1", "ref_2", "ref_3", "ref_4", "ref_5",
              "judgment (correct | partial | incorrect | nonsensical)", "note (optional)"]
    for form in ("A", "B"):
        with open(HV / f"annotation_form_{form}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(fields)
            for i, it in enumerate(chosen, 1):
                r = (it["refs"] + [""] * 5)[:5]
                w.writerow([i, it["image_url"], it["question"], it["model_answer"], *r, "", ""])

    key = [{"id": i, "category": it["category"], "token_f1": it["f1"],
            "model_answer": it["model_answer"], "refs": it["refs"]}
           for i, it in enumerate(chosen, 1)]
    cat_counts = dict(sorted(Counter(it["category"] for it in chosen).items()))
    (HV / "answer_key.json").write_text(json.dumps(
        {"model": a.model_label, "pred_file": str(Path(a.pred).relative_to(ROOT)),
         "n": len(chosen), "category_counts": cat_counts, "items": key},
        ensure_ascii=False, indent=2))

    f1s = [it["f1"] for it in chosen]
    partial = sum(1 for x in f1s if 0.2 <= x <= 0.6)
    (HV / "ANNOTATION_README.md").write_text(f"""# Human validation — annotation instructions

**Model under review:** {a.model_label}
**Task:** judge whether the model's answer to each question is right, using the
image and the five reference answers.

This is the 2-annotator replacement for the earlier single-rater self-check
(`selfcheck_judgments.json`, N=120, no image). Same model, same rubric — now
with the image and a second independent rater so we can report Cohen's kappa
and check how well token-F1 tracks human judgment.

## Who does what
- **Two annotators, working independently** — do not discuss items while
  annotating. Annotator 1 fills `annotation_form_A.csv`, annotator 2 fills
  `annotation_form_B.csv`. Both files have the **same {len(chosen)} rows**.
- When both are done: `python scripts/human_validation_report.py` prints the
  per-label rates, Cohen's kappa, and the token-F1 vs. human breakdown.

## How to fill a row
1. Open `image_url` in a browser to see the image.
2. Read `question`, the model's `model_answer`, and `ref_1..ref_5`.
3. Put exactly one of these in the **judgment** column:

| label | meaning |
|---|---|
| `correct` | right for the image + question (phrasing may differ from the references) |
| `partial` | right topic/object but wrong or missing in a detail (wrong count, one of two attributes off, over-broad) |
| `incorrect` | wrong object / attribute / relation, or answers a different question |
| `nonsensical` | not a coherent answer (word salad, empty, pure repetition, unrelated) |

4. Optional `note`: a few words if the item is ambiguous or the references
   themselves disagree with the image.

## Rubric notes
- References are free-form and diverse; the model does **not** need to match
  their wording. Judge meaning against the image.
- Open questions (causal / context / action) often have several valid answers:
  mark `correct` if the model's answer is a plausible answer to that question
  for that image, even if no reference states it exactly.
- Cannot load the image (404)? Leave `judgment` blank, note "no image".

## Sample
{len(chosen)} questions, proportionally stratified by AutoViVQA reasoning-type
category:

{chr(10).join(f'- {c}: {n}' for c, n in cat_counts.items())}

Token-F1 in the hidden key spans the full range; {partial} of the {len(chosen)}
fall in the 0.2-0.6 "partial-overlap" band that the self-check flagged as only
~43% semantically acceptable — the main thing this study re-checks.
""")
    print(f"wrote {len(chosen)} items -> {HV.relative_to(ROOT)}/")
    print("  annotation_form_A.csv, annotation_form_B.csv, answer_key.json, ANNOTATION_README.md")
    print("categories:", cat_counts)


if __name__ == "__main__":
    main()
