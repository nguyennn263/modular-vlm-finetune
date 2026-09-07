# Human validation — annotation instructions

**Model under review:** multi_token bridge, frozen backbone, seed 42 (plain, 2 epochs)
**Task:** judge whether the model's answer to each question is right, using the
image and the five reference answers.

This is the 2-annotator replacement for the earlier single-rater self-check
(`selfcheck_judgments.json`, N=120, no image). Same model, same rubric — now
with the image and a second independent rater so we can report Cohen's kappa
and check how well token-F1 tracks human judgment.

## Who does what
- **Two annotators, working independently** — do not discuss items while
  annotating. Annotator 1 fills `annotation_form_A.csv`, annotator 2 fills
  `annotation_form_B.csv`. Both files have the **same 300 rows**.
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
300 questions, proportionally stratified by AutoViVQA reasoning-type
category:

- Câu hỏi có/không: 3
- Lý do/ Nhân quả: 38
- Mô tả hành động: 21
- Mô tả thuộc tính: 1
- Mô tả vị trí/ không gian: 44
- Mối quan hệ: 90
- Suy luận ngữ cảnh: 8
- Xác định số lượng: 38
- Xác định thuộc tính: 2
- Xác định đối tượng/ thuộc tính: 55

Token-F1 in the hidden key spans the full range; 157 of the 300
fall in the 0.2-0.6 "partial-overlap" band that the self-check flagged as only
~43% semantically acceptable — the main thing this study re-checks.
