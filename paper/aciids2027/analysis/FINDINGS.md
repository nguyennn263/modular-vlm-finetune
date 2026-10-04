# Analysis findings for the ACIIDS 2027 paper (session 2026-10-04/05)

All numbers ×100, AutoViVQA grouped split, scorer `metrics.vqa_metrics` (same as Table 2),
whole-image evaluation unless marked "crop". Predictions used: `mt-s42-t1-full`,
`l3ep-s42-t1-full` (input_diag), `regen_full/l1ep-s42`; crop = old first-448px-tile eval.

## 1. How much does the model use the image? (seed 42, val)

| Probe | Result |
|---|---|
| Question-only baseline (nearest train question by TF-IDF, copy its answer #1) | F1 **36.17** (ViBridge 3ep: 55.04) |
| Answers unchanged, crop (1/6 of image) vs whole image — +LoRA 3ep | **80.6 %** (F1 54.91 → 55.04) |
| same — bridge only | 63.4 % (F1 49.61 → 50.87) |
| ... by category (+LoRA) | counting 93.0 %, yes/no 90.9 %, action 85.7 %, recognition 79.7 %, spatial 78.7 %, relational 77.9 %, causal 76.3 %, context 70.3 % |
| Answers that DO change crop→full (1061): better / worse / tie | 439 / 385 / 237 (≈ random) |
| Same question asked about >1 image (112 questions) | identical answer for all images in **84 %** |
| Predictions that are verbatim training answers | 57.3 % (+LoRA), 51.2 % (bridge); references: 48.4 % |
| Most frequent +LoRA answers | "màu trắng" 100, "hai chiếc" 91, "màu xanh" 86, "đang gặm cỏ" 85, "hai con" 82 |

Reading: the CLS-only bridge passes scene-level gist; most of the gain is answer style /
language prior. Cause: `pooler_output` = `last_hidden_state[:,0]` (CLS), which Vintern
discards (`extract_feature`: `vit_embeds[:, 1:, :]`); Vintern feeds 256 pixel-shuffled patch
tokens per 448px tile through `mlp1`.

## 2. Per reasoning type, bridge → +LoRA 3ep (seed 42, val, whole image)

relational 55.71→60.49 (+4.8) · recognition 45.32→49.65 (+4.3) · spatial 47.57→51.95 (+4.4) ·
causal 41.26→48.63 (+7.4) · counting 66.57→66.69 (+0.1) · action 44.38→47.21 (+2.8) ·
context 35.10→37.26 (+2.2) · yes/no 64.41→69.40 (+5.0). Overall 50.87→55.04.
Counting: LoRA fixes 37, breaks 29 (F1 threshold 0.5); causal fixes 140, breaks 58.
Mean reference length (words): relational 5.6, causal 5.0, counting 2.8.
3.7 % of all questions (mostly counting) ask about the generating caption ("được nhắc đến
trong mô tả"); LoRA gain on them +0.3.

## 3. Paired bootstrap (val, seed 42, 2000 resamples, whole image)

LoRA 1ep − bridge: +2.78 [2.20, 3.35]; LoRA 3ep − bridge: +4.17 [3.55, 4.79]; P(Δ>0)=1.000.

## 4. Tile diagnostic (input_diag, Multi-Token seed 42, val, whole image)

| Input to bridge | 1 tile | 3 tiles | 6 tiles |
|---|--:|--:|--:|
| CLS (mean of per-tile CLS) | 50.87 (CE 1.489) | 50.84 (CE 1.491) | 49.97 (CE 1.509) |
| mean of every token (old T>1 path) | 18.32 (CE 3.468) | 18.31 (CE 3.469) | 18.96 (CE 3.419) |

The old "collapse beyond one tile" (F1 21.05) was an artifact of switching CLS → token
mean at T>1; with a consistent CLS input more tiles neither help nor hurt. Caveats: on 4:3
val images a 3-tile request tiles 1×1 and repeats the image (no new content); 6 tiles =
six 448px crops, no thumbnail. The old tile-augmentation training (mixed CLS/mean inputs)
is invalid and is not used in the paper.

## 5. Verified bugs / corrections

- **Full Q-Former answer leak**: training `input_ids` contain `assistant\n{answer}`
  (decoded from the real collator); `trainer.py` passes their embeddings to the `qformer`
  bridge on every branch. Fixed by the peer session in 4a8eb6d (exp/eval-input-diagnostic);
  retrain pending. Light Q-Former is not text-conditioned.
- **"+LoRA" is one joint run, not stage 2**: cosine(plain bridge, bridge inside LoRA ckpt)
  = 0.001–0.002; LoRA-1ep `global_step` = 3222 = 1 epoch from 0.
- Multi-Token's 8 projection matrices are pairwise near-orthogonal (cos ≤ 0.02).
- Training time (2 ep bridge): Multi-Token 4.93 h, Light Q-Former 5.01 h; LoRA 1 ep 2.63 / 2.67 h.

Scripts: `by_cat.py` (per-category), `qf_leak_test.py` (leak probe; needs `einops`, `timm`).
