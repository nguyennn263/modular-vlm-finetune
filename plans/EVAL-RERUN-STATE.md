# Full-image re-evaluation — working state (2026-10-04)

Read this first to resume. Results themselves: `plans/results-full-image.md`
(auto-regenerated + committed by `scripts/parallel/collect_loop.sh`).

## Why this re-run exists

1. **First-tile crop bug.** At `n_tiles=1`, generation (`trainer._load_pixels_for_generation`)
   fed the first 448px crop of the dynamic tiling = top-left 1/6 of every val image (all 2909
   val images tile 3x2). Training and val loss saw the whole image at 336px. Every reported
   F1/BLEU/ROUGE/METEOR/CIDEr was measured on the crop. Fix: `--gen-image full`.
   Re-scoring only. Checkpoints are fine: val loss reproduces to 3 decimals.
2. **CLS vs mean pooling (RQ3).** Pooled bridges (multi_token, residual) read InternViT's CLS
   token at 1 tile but the mean of all tokens at T>1 (`setup.py`, `trainer._vision_embeds`).
   `--pooled-input {mean_all, cls_mean}` separates the two.
3. **Metric unification.** `metrics/compute_score.py` (f1_token = character-level F1, wup broken,
   per-sample CIDEr = 0) is deleted on this branch. Every number now comes from
   `metrics.vqa_metrics.score_answers` (the trainer's scorer; reproduces stored eval JSONs exactly).

## Decisions made by the user

- Re-measure with full image: tab:main/stability (val+test), RQ3, tab:levers (RQ5),
  tab:bridges (plain AND +LoRA 1ep, all seeds), tab:ood.
- NOT re-run: num_tokens sweep / Patch-Pool / Conv-Abstractor (advisor-only §6.2), RQ6 LoRA
  placement, Vintern-FT reproduction, blank/shuffled image ablation (never requested).
- One metric implementation project-wide (vqa_metrics); OOD F1 uses it too.
- Local-only checkpoints go up as PRIVATE Kaggle datasets on the running account (no public upload).
- Up to 2 kernels per Kaggle account (Kaggle caps GPU sessions at 2; verified 2026-10-04).
- **Found 2026-10-05: Full Q-Former leaks the answer during training** (collator puts the answer in
  input_ids; trainer feeds those embeddings to the qformer bridge; unmasked cross-attention; bridge
  tokens precede the text). Its checkpoints are compromised; re-eval can't fix them. Rows flagged
  "LEAK" in the summary. **Pending user decision:** drop Full Q-Former (recommended; use Light
  Q-Former as capacity evidence) vs fix + retrain.
- **Pending user decision:** RQ3 tile-augmentation row (trained with mixed CLS/mean input):
  drop it (recommended) vs retrain.

## Results so far (headline)

- OOD (final, 12/12): ours ≫ Vintern only on ViVQA; ViVQA-X on par (ours +1.4 F1, Vintern +8 CIDEr);
  Vintern far ahead on text-reading sets. Full image vs crop changes our OOD F1 by < 0.5.
- Plain Multi-Token: +1.0–1.4 F1 with full image (val and test). LoRA 1ep/3ep: ≈ +0.0–0.3.
  So the LoRA gain (RQ6) shrinks by ~1 point.
- RQ3: 1 tile + mean of tokens = F1 18.32 (loss 3.47), same as old "3 tiles" (18.31). 3 tiles on
  4:3 images = the whole image repeated 3x (grid 1x1 padded). 6 tiles with mean of per-tile CLS =
  F1 49.97 (no collapse). The old RQ3 conclusion is wrong: the collapse is the CLS→mean switch.

## Where things are

- Branch `exp/eval-input-diagnostic` (pushed). Worktree: scratchpad `wt-diag/` of session 7534ec64.
- Ledger: `outputs/parallel/ledger.json` (keys `input-diag:*`, `ood-full:*`, `regen-full:*`;
  `-r2` = re-run of 4 diag jobs stuck beside their Python 3.13 first launch).
- Launch: `scripts/parallel/paper_queue.py fill --bridges` (all jobs launched 2026-10-04 23:2x).
- Collect: `scripts/parallel/collect_loop.sh` (every 10 min: collect, regenerate summary, commit, push).
- Kaggle image must be pinned (`DOCKER_IMAGE` in `input_diag.py`); default image is Python 3.13
  and breaks `setup_kaggle.sh`.

## Still to do after all jobs land

1. Aggregate per table (mean ± std), send to the other session (owns `paper/aciids2027/`).
2. Update the advisor report `plans/paper-status-for-advisor.md` (main checkout, uncommitted edits):
   new numbers, add §6.0 explaining the crop bug, rewrite RQ3, RQ6 Δ.
3. Merge `exp/eval-input-diagnostic` into the paper branch; there, delete `compute_score.py` and
   point `experiments/vintern-ft/score_local.py` to `score_answers` (or drop Vintern-FT scripts).
4. Mark stale docs (P6-draft, results-*.md, old paper draft) as superseded.
