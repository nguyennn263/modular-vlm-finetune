# Reproducing the "Vintern-1B (fine-tuned)" baseline

**Goal.** Re-verify the AutoViVQA `Vintern-1B (fine-tuned)` row (Acc 13.01 / F1 53.76
/ CIDEr 72.84) by running Vintern's **own official fine-tune cookbook** on **our
grouped leak-free split**, so the number in our Table 1 is one we measured, not
one we cite.

Branch: `exp/vintern-ft-baseline`. Decisions (2026-09-10): grouped split · follow
the cookbook verbatim · 1 seed sanity first.

---

## ⚠️ What the cookbook recipe actually is

Source: <https://colab.research.google.com/drive/1Iu6ssNvlA2bNGlOLq7xR0l-5qtBIqul4>
(saved verbatim in `reference/`), which is the `5CD-AI/Vintern` repo's
`internvl_chat` 2nd-stage fine-tune (`reference/repo_default_finetune_lora.sh`).

The recipe is **NOT** "train all of InternViT + projector + LoRA the LLM". It is:

| knob | cookbook value |
|---|---|
| `freeze_backbone` | **True** (InternViT frozen) |
| `freeze_mlp` | **True** (projector frozen) |
| `freeze_llm` | **True** |
| `use_llm_lora` | **16** (LoRA on Qwen2.5-0.5B only ≈ 2.2 M params) |
| `max_dynamic_patch` | **6** tiles (not 12) |
| `force_image_size` | 448, `down_sample_ratio` 0.5, `use_thumbnail` True |
| optimiser | lr 4e-5, cosine, warmup 0.03, wd 0.01 |
| schedule | **1 epoch**, total batch 16, `max_seq_length` 700 |
| template | Hermes-2 |

**Implication for our paper's framing.** The "Vintern fine-tune = ~100× our
training cost / full-backbone" line describes how 5CD-AI *built*
Vintern-1B-v3_5 from InternVL2.5-1B (millions of pairs, full stack). The
*downstream fine-tune* that a practitioner runs — and most likely what AutoViVQA
reports — is frozen-backbone + LoRA-16-LLM + 6 tiles + 1 epoch. That is
**roughly our own adaptation budget** (~2.2 M LoRA params vs our bridge 7.35 M +
decoder-LoRA 2.16 M). Our real differentiators then are: (1) we replace the
frozen projector with a trained bridge, (2) **1 tile vs 6** (≈4–6× cheaper vision
inference), (3) the 6-axis bottleneck diagnosis. The reproduction here tells us
which recipe AutoViVQA used: if this lands near F1 53.76, they used the cookbook.

---

## Pipeline

1. **`build_data.py`** — `data/splits/{train,val,test}.jsonl` → InternVL chat
   SFT JSONL (`data/autovivqa_{split}.jsonl`) + `data/meta_autovivqa.json`.
   Answer target = `answers[0]` (matches bridge `answer-sampling=first`).
   Already run; outputs committed.
2. **`finetune_lora.sh`** — the cookbook shell, verbatim hyper-params, paths +
   GPU-count parametrised. `GPUS`, `MODEL_PATH`, `META_PATH`, `OUTPUT_DIR`, `SEED`
   via env.
3. **merge LoRA** — `reference/` cell 41 (`tools/merge_lora.py` from the Vintern
   repo) then copy `*.py` + `config.json` from the base model (cookbook cells
   45/47).
4. **`gen_vintern.py`** — greedy `InternVLChatModel.chat` over val + test at
   `max_num=6`, writes `text_predictions_epoch_1.json` (bridge-pipeline format).
   Generation only — no metric deps in the kernel (stays in the transformers
   4.47 / InternVL training env).
5. **`score_local.py`** (run after fetch, locally) — `metrics.compute_score.
   compute_all_data` (same function behind Table 1) for the in-house 8 metrics +
   corpus CIDEr-D / BLEU-4 / ROUGE-L. This keeps the number apples-to-apples with
   our bridge rows.

## Kaggle

- Images: existing dataset `nguynrichard/auto-vqabest`
  (`/kaggle/input/auto-vqabest/preprocessed_images/`, original resolution).
- `run_kaggle.py` builds + pushes the worker notebook (clones this repo + the
  Vintern repo, pip-installs per cookbook, downloads `5CD-AI/Vintern-1B-v3_5`,
  trains, merges, evals, leaves everything under `/kaggle/working`).
- Cost estimate (pre-run): LoRA-only, 6 tiles, 1 epoch over 25.9k samples ≈
  3–5 h on P100; eval ≈ 2–4 h.
- **Actual (2026-09-10, acc15): the estimate was wrong.** With 6 tiles + eager
  attention (Kaggle P100/T4 are pre-Ampere → no flash-attn), training ran at
  ≈38 s / optimizer step. Over 12 h it reached **step 1000 / ~1611** (≈62 % of
  epoch 1) and was killed by the Kaggle 12 h cap before generation started. A
  full epoch would need ≈17 h + ≈6 h generation ≈ **23 h — not doable in one
  Kaggle session.** ~17.8 h of acc15's 30 h quota spent (5.5 h of that on a
  wasted flash-attn source build in the first attempt; env debugging took 6
  kernel iterations).
- **Verdict: the cookbook recipe as-written cannot be reproduced end-to-end in
  one Kaggle session.** Options: (a) resume from the mid-training checkpoint;
  (b) rerun at `--max_dynamic_patch 1` (≈6× faster, and a "Vintern-FT @ 1 tile"
  number is apples-to-apples with our 1-tile bridge); (c) subsample train to
  ≈8k; (d) defer — the paper's framing fix does not depend on this number.

## Comparison target (our numbers, grouped split, for reference)

| | F1 | CIDEr(ih) | CIDEr-D |
|---|--:|--:|--:|
| our bridge (1 tile, 0.78%) | 49.55 | 96.49 | 92.3 |
| our bridge + decoder-LoRA (1 ep) | 53.52 | 106.56 | 103.2 |
| our bridge + decoder-LoRA (3 ep) | 54.71 | 110.49 | 107.5 |
| AutoViVQA "Vintern-1B (fine-tuned)" (cited) | 53.76 | 72.84 | — |
| **this experiment (cut at step 1000/1611, no generation)** | _no result — see above_ | — | — |
