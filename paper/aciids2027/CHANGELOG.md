# Paper changelog (ACIIDS 2027, ViBridge-VQA)

## 2026-10-09 (evening) — OOD evaluation and global–local + decoder LoRA

- **OOD table back** (Table 5, §4.7): ViTextVQA, OpenViVQA, ViVQA-X, ViVQA × sampling seeds
  42/123/3407; Vintern-1B zero-shot (6 tiles), global only + LoRA, ours k=36 / k=144 (seed-42
  checkpoints). k=144 exceeds six-tile Vintern-1B on ViTextVQA (33.26 vs 32.01 F1); Vintern-1B
  stays better on OpenViVQA. Source: `analysis/ood_gl_table.json` (`score_gl.py` @991fb21).
- **ViBridge-VQA + decoder LoRA** (g14-k36 + rank-16 LoRA, 1 joint epoch, 3 seeds; 15.02M =
  1.57% trained): test F1 58.17 ± 0.19, CIDEr 122.88 — best accuracy / BLEU / ROUGE-L /
  METEOR / CIDEr of all systems; ViMoE-VQA keeps P and F1 (+2.5). Added to Tables 2–3,
  Fig. 2 (a: star; b: hatched bars), §4.2, ablation, reasoning types, abstract, intro,
  conclusion. Source: `gl_results.json["g14-k36+lora1"]` (@9db84ba).
- Claim changed: decoder LoRA is "not needed to surpass global-only + LoRA, but
  complementary" (+3.3 val F1 on top of the global–local bridge vs +2.7 on global only;
  best in 7/8 reasoning categories).
- Space: Table 1 (data) and the global-bridge design table folded into text, ViT tile-cost
  table into the Efficiency paragraph. Still 15 pages.

## 2026-10-09 — rewritten around the global–local bridge

**Why.** The global–local bridge (Multi-Token on `[CLS]` + Vintern's frozen `mlp1` tokens,
pooled, in Vintern's image slot) beats every earlier configuration without decoder LoRA.
Study: `plans/global-local-results.md` on `exp/eval-input-diagnostic`.

**Method now.** Frozen ViT + frozen projector (`mlp1`) + frozen Qwen; only the global
bridge trains (g=14: 12,857,600 params, 1.35% per the training log). Local tokens:
`mlp1` output on the 12×12 grid at 336px, average-pooled to k ∈ {1, 9, 36, 144}, written
into `<img><IMG_CONTEXT>…</img>`. Code: `GlobalLocalBridge` (commits e9620fe, 84dd815).
Selected budget g=14, k=36 = knee of the accuracy–cost Pareto front.

**Where every number comes from.**
- ViBridge-VQA rows, Table 4 (budget), Fig. 2: `analysis/gl_results.json`, written by
  `analysis/gl_results.py` from `exp/eval-input-diagnostic` @16dac85
  (`outputs/train_gl/`, 24 runs = 8 configs × seeds 42/123/3407).
- Global-only (Multi-Token, + LoRA) rows, Table 6 (bridges): whole-image re-evaluation,
  `plans/paper-tables-full-image.md` (2026-10-05); Full Q-Former = leak-fixed retrain.
- Table 7 (global bridge design): earlier 448px-crop evaluation, stated in its caption.
- Fig. 3 examples: seed-42 val predictions, `figures/make_qualitative.py` header.
- Trainable %: training logs (`Trainable parameters: 12,857,600 (1.35%)`).

**Changed.**
- Title, abstract, introduction, contributions, method (new Fig. 1), conclusion.
- Table 2: added the two ViBridge-VQA rows; old "ViBridge-VQA" rows relabelled
  "Global only" (Multi-Token, + decoder LoRA).
- Table 3 (stability): global only, + LoRA 3 ep, ours k=36, ours k=144.
- New Table 4 (token budget) and §4.3 (selection: knee, TOPSIS, parity with LoRA).
- Fig. 2: (a) F1 vs visual tokens with the knee; (b) by reasoning type, ours vs global
  only vs + LoRA. Fig. 3: new examples (3 where local tokens fix the answer, 1 failure).
- Related work: token-budget paragraph; refs added: LLaVA-1.5, Matryoshka, Kneedle, TOPSIS.

**Removed.**
- Levers table (random reference, distillation): now two sentences in §4.5.
- Out-of-distribution table: it evaluated the Multi-Token + LoRA model; the global–local
  model has not been evaluated OOD yet (named as future work in §5).
- Claim "decoder adaptation is the key ingredient": the frozen-decoder global–local model
  beats every decoder-LoRA model; decoder LoRA now reads as compensating for poor
  visual tokens.

**Open before submission.**
- OOD evaluation of ViBridge-VQA (k=36), especially ViTextVQA / OpenViVQA.
- Global–local + decoder LoRA (per-category results suggest they are complementary).
- 15 pages = the CFP maximum.
