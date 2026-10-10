# Paper changelog (ACIIDS 2027, ViBridge-VQA)

## 2026-10-11 (evening) — pre-submission review fixes

Review points checked against the raw runs; all fixed in text, captions and figures.

- **Numbers verified.** `analysis/verify_numbers.py` re-scores all 54 ViBridge-VQA runs
  (9 configurations x 3 seeds x val/test) from the prediction files with `score_answers`:
  max |diff| vs the Kaggle eval JSONs and vs `gl_results.json` = 0.0000. Global-only rows: the
  per-seed eval JSONs reproduce 50.74 (4 seeds) and 54.78 (3 seeds).
- **Fig. 2b vs §4.6 (real inconsistency).** The figure showed ours+LoRA minus global only (seed
  42), the text quoted k=36 minus global only (seed 42) — different comparisons. Now every model
  is averaged over all its seeds (`analysis/per_category.py` -> `per_category.json`; global only
  4 seeds, others 3) and both the figure labels and the text give ViBridge-VQA (k=36, no LoRA)
  minus global only: recognition +11.0, action +6.4, spatial +5.8 (was +10.2/+5.4/+5.1 vs seed
  42). Error analysis also uses all seeds (`error_analysis.json`; F1=1 for global only 9.7%).
- **Token-budget selection.** Method and §4.4 now say the knee and TOPSIS are computed on
  validation only and the test knee is a post-hoc check. Log-axis claim corrected: on
  validation the log-axis knee is g14-k36 for 4/8 metrics and g8-k9 for the other 4 (old text:
  "11 of 16 cases", which mixed val and test).
- **Trainable parameters.** Defined once (Method §3.4): % of all model parameters = frozen
  Vintern-1B components (~0.94B incl. vision encoder, projector, embeddings, output layer) +
  trained modules. Bridge = 1025*896*g parameters (12,857,600 for g=14); LoRA = 2,162,688;
  totals 1.35% / 1.57% check out against the training logs (frozen 934,628,608 without mlp1,
  + mlp1 4,482,816).
- **Cost (real error).** The old text attributed 362 GFLOPs / 229 ms to one 336x336 view; that
  profile (plans/final-plan.md, P1) was one 448x448 tile, and 2,172 GFLOPs was 6 tiles without
  the thumbnail. Now: decoder tokens (50 vs <=1,792), vision-encoder GMACs from
  `analysis/vit_cost.py` (191 per 336 view vs 362 per 448 view, <=2,533 for 7 views = 13x;
  matches fvcore at 448), and evaluation runtime (Table 2: session incl. loading, greedy, batch
  2, Kaggle P100/T4). No latency speed-up claimed.
- **Baselines on another split.** Caption of Table 1, Baselines paragraph, abstract,
  intro and conclusion call them "published" reference points, not a controlled comparison.
- **Abstract/intro/conclusion** separate bridge only (50 tokens, 1.35%, F1 54.6) from bridge +
  LoRA (50 tokens, 1.57%, F1 58.2). Intro states the gap (global-only bridge vs learned patch
  compressor vs reusing the pre-trained alignment) and the research question; Related Work
  contrasts mechanism / parameters / budget selection; conclusion adds the scope limit
  (Vintern-1B, tested benchmarks; global tokens only compared with g=8).
- Claims limited to the tested configurations ("within the tested ranges"); no g=0 run exists.
- Dropped for space: preliminary pooling-operator sentence, five redundant DOIs. 15 pages.

## 2026-10-11 — prose rewrite after AutoViVQA (ACIIDS 2026) and ViMoE-VQA (KES 2026)

**Why.** The user found the text machine-like. Rewritten in the group's style: narrative abstract
(context -> gap -> method -> outcome, few numbers), intro ending in "To address this question,
we propose..." + three bold contributions, prose Related Work (no paragraph headings), Method
opening with "Overall Architecture", Experiments with interpretive paragraphs instead of
lists of deltas, a stability paragraph, error analysis, three-paragraph conclusion
(summary / limitations / outlook). No result changed.

**Tables.**
- Table 1 (main): ± std only on our rows (as ViMoE-VQA does); the old stability table
  (val+test, 8 metrics) removed — val-test gap and max std now in the "Statistical stability"
  paragraph.
- Table 2 (budget): ± only on F1; val/test grouped.
- Table 3 (was bridges, 13 columns): now "where the visual tokens come from" — F1/CIDEr bridge
  only vs + decoder LoRA, with ViBridge-VQA k=36/144 as the last group (2x2 view of local
  tokens x LoRA). k=144 + LoRA not trained ("–").
- Table 4 (OOD): datasets as column groups, F1 (± std) and CIDEr only.
- Global-only + LoRA (3 ep.) in Tables 1 and 4 is stated as "the strongest global-only model,
  as a reference" (answers the LoRA/no-LoRA asymmetry question; option (b)).

**New numbers (error analysis, §4.6).** `analysis/error_analysis.py` -> `error_analysis.json`
(val; global only / + LoRA seed 42, ours 3 seeds): answer length 4.3 words = reference length;
77–81% partial overlap; F1=1: 9.8% -> 13.3% (k=36); no overlap: 9.6% -> 8.0% (6.8% + LoRA);
no-overlap concentrated on context/recognition/spatial/action (12–18%).

**Claims tightened.** Abstract/conclusion say "highest accuracy, ROUGE-L and METEOR" (not
"best generation scores", because of the BARTPhoBEiT BLEU/CIDEr outliers); main text says
"setting aside the outlying BARTPhoBEiT values". "Global bridge design" paragraph folded into
§3.2 (one sentence); the random-reference lever dropped. Two bib entries shortened. 15 pages.

## 2026-10-10 — OOD of ViBridge-VQA + decoder LoRA

- Table 5: row "ours, k=36 + LoRA" on the four OOD sets (seed-42 checkpoint, same sampled
  questions). F1: ViTextVQA 15.39, OpenViVQA 28.99, ViVQA-X 27.64, ViVQA 44.76 — LoRA helps the
  general-domain sets (+2.7 / +2.1 over k=36) but hardly scene text (+0.3 / +0.8). Text of §4.7
  and the conclusion updated. Source: `analysis/ood_gl_table.json` (`score_gl.py --gl-lora`,
  @8c980a2).
- Qualitative and global-bridge-design paragraphs shortened to stay at 15 pages.

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
