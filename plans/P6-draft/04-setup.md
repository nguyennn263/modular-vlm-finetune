# 4. Experimental Setup

## 4.1 Dataset and split

AutoViVQA provides 19,411 images / 37,077 Vietnamese questions, each with five
diverse free-form answers and a reasoning-type label, released with an 80/20
train/val division and **no public test split**. We re-partition the data into a
**grouped 70/15/15 split**: images are assigned to exactly one of
train / val / test, so no image (and therefore no shared caption or scene
context) crosses splits. Assignment is by hashed `image_id`, seed 42, with the
category distribution held approximately fixed across splits.

| Split | Questions | Images | Ans/q | Ans length | Question length |
|---|---:|---:|---:|---:|---:|
| Train | 25,933 | 13,576 | 5.00 | 4.3 w | 11.4 w |
| Val | 5,544 | 2,908 | 5.00 | 4.3 w | — |
| Test | 5,503 | 2,914 | 5.00 | 4.3 w | — |

Image overlap between any two splits is **0**.

Reasoning-type (`category`) distribution is stable across splits; the dominant
classes are `relational` (~30 %), `recognition` (~19 %), `spatial` (~15 %),
`causal` (~13 %), `counting` (~12 %), with `action`, `context`, `yes/no` in the
tail. The nominal `reason_depth` field (Level 1–5) is retained for reference
only; we use the nominal `category` labels throughout and treat them as
unordered (Level 5 has only ~200 train examples and is not a reliable "hardest"
tier).

**Oracle-sweep subset.** Running every action on every question is expensive
(§4.4). For the oracle sweep we use an **equal-per-category subset** (cap
≈ 625 questions per category), giving 5,547 train / 3,727 val / 3,739 test
questions actually swept. This is a deviation from proportional stratification —
it over-samples the tail categories — and it is accounted for when aggregating
oracle statistics (per-category means are unweighted; overall means are
re-weighted to the natural distribution).

## 4.2 Answer selection during training

Each question has five references. Bridge training optimises cross-entropy
against the **first** reference (a fixed choice, not a random draw, for
reproducibility). All **evaluation** metrics use all five references
(per-sample max for token metrics; corpus multi-reference for CIDEr-D / BLEU /
ROUGE / METEOR). An `answer_sampling ∈ {first, random, majority}` switch exists
in the trainer but all reported runs use `first`.

## 4.3 Bridge training

InternViT-300M and Qwen2-0.5B frozen; only the bridge trains. AdamW, batch size
8, 4 epochs, no early stopping (validation CE bottoms out within one epoch while
CIDEr keeps climbing for ~2 epochs — early stopping on CE truncates training
prematurely). A checkpoint is saved every epoch; the epoch-4 weights are used for
final evaluation. The auxiliary distillation term (0.5·MSE between the bridge's
first token and a frozen linear-projector baseline) that appeared in an earlier
version is **disabled** for all runs here — it inflates the reported loss without
affecting decoding, and applied to only two of the five bridges, an unfair
confound.

Metrics converge by epoch 2 (multi-token CIDEr 1.025 at epoch 2 → 1.029 at
epoch 4 on an 800-sample probe); §5.1 numbers are epoch-1 full-val, a slight
underestimate.

## 4.4 Hardware, compute budget, reproducibility

All training and the oracle sweep ran on Kaggle P100 (16 GB) and T4 GPU kernels
(12 h wall-clock cap per kernel). Approximate GPU-hours by category
[**TODO: final tally from kernel logs**]:

| Item | GPU-h (approx) |
|---|---:|
| Bridge training (5 bridges × ~10 h, seed 42) | ~52 |
| Oracle sweep — val (\|A\|=6, 5 shards) | ~20 |
| Oracle sweep — val/test/train (\|A\|=9 multi-token, 15 shards) | ~45 |
| Router + f(I,Q) + policy training | ~4 |
| Failed / re-run tile-augmented training | ~36 |
| **Total** | **~200** |

- **Seeds.** Bridge and policy runs: seed 42. Split: seed 42. A 5-seed protocol
  (42, 123, 3407, 2026, 8668, matching ViMoE-VQA) is planned for the multi-token
  row and the policy ladder; single-seed results are supported by paired
  bootstrap over the evaluation split.
- **Libraries.** PyTorch 2.x, Transformers, `pyvi` tokeniser for Vietnamese,
  PhoBERT-base-v2, InternViT-300M / Qwen2-0.5B from Vintern-1B-v3.5. Metrics:
  pycocoevalcap-style corpus CIDEr-D / BLEU / ROUGE / METEOR (METEOR via the
  Meteor 1.5 Java jar); token-level P/R/F1 from an in-house scorer.
- **Statistics.** Main comparisons use paired bootstrap (10k resamples) over the
  evaluation split; the test split is scored exactly once.

## 4.5 Metrics

Generation quality: **corpus CIDEr-D** (primary — most implementation-stable
across papers), corpus BLEU-4/BLEU-1, corpus ROUGE-L. METEOR is reported for
completeness but not used for cross-paper claims (it varies by ~13 points
between implementations on identical predictions). Answer matching: token-level
Precision / Recall / F1 and exact-match / WUPS, per-sample max over the five
references. For the oracle, `M(a; x)` = per-sample CIDEr against the five
references.
