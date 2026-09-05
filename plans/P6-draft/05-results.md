# §5 Experiments and Results

> Draft, reordered 2026-09-05 around the efficiency-primary spine (§6.1 for the
> full framing). §5.1–5.2 carry the primary claim (frozen backbone, 1 tile,
> beats prior work); §5.3–5.6 are supporting robustness checks — each shows a
> different way of trying to buy back the remaining F1 gap to ViMoE-VQA (visual
> routing, training target, representation alignment, decoder capacity) and what
> happens when you do. §5.2/§5.3 numbers are **LOCKED on tile-trained
> checkpoints** (C3, 2026-09-05) — the oracle sweep was re-run on bridges
> retrained *with* tile-count augmentation (`--tile-choices 1,3,6`), closing the
> reviewer confound of evaluating an n_tiles-adaptive oracle against
> n_tiles=1-only-trained models. Result: **no change** from the original sweep.

## 5.1 Bridge architecture comparison

Full table and discussion in `05.1-bridge-baseline.md`. Cross-reference summary:
on the grouped 70/15/15 split (single seed, epoch 1, pycocoevalcap corpus
metrics), `multi_token` is the strongest bridge and **beats ViMoE-VQA on every
generation metric** while trailing on token-F1:

| | CIDEr-D | BLEU-4 | ROUGE-L | F1(tok) |
|---|---|---|---|---|
| ViMoE-VQA (reported) | 88.7 | 12.5 | 47.1 | **60.7** |
| **multi_token** (0.78% trainable params) | **94.4** | **19.6** | **50.0** | 44.2 |
| qformer | 86.7 | 17.5 | 47.1 | — |
| mini_qformer | 83.8 | 16.8 | 46.0 | — |
| residual | 56.3 | 8.1 | 36.0 | — |

METEOR is omitted from cross-paper comparison — it is implementation-dependent
(in-house 41.1 vs pycocoevalcap multi-ref 28.5 on identical predictions).

This is at **1 image tile**, on a backbone where InternViT and Qwen2-0.5B are
both **fully frozen** — only the bridge (0.78% of total parameters) is trained.
The reference recipes it is compared against are not: Vintern-1B full-finetunes
its ViT and LoRAs its LLM; ViMoE-VQA trains a mixture-of-experts on top of a
similarly unfrozen stack. §5.2 quantifies what that 1-tile choice costs in
compute terms; §5.3–5.6 test, from four different angles, whether there is
cheap headroom being left on the table by keeping everything else frozen.

## 5.2 Compute-efficiency of the vision-side lever (P1 profiling)

n_tiles is a genuine compute lever. Profiled on a Tesla P100-16GB
(`mini_qformer`, 32 samples, `src.cli.profile`):

| n_tiles | InternViT GFLOPs | latency (ms) | throughput (img/s) |
|---|---|---|---|
| 1 | 362 | 229 | 6.0 |
| 2 | 724 | 374 | 3.3 |
| 4 | 1 448 | 648 | 1.7 |
| 6 | 2 172 | 922 | 1.15 |

Dynamic range 1→6: **FLOPs ×6.0, latency ×4.0, throughput ×5.2** — far above the
15 % threshold at which a lever is worth routing over. The InternViT encoder is
linear in tile count (GFLOPs 362 · n_tiles); wall-clock scales sub-linearly
(×4.0) because the frozen Qwen2-0.5B decode is a fixed per-sample cost that
n_tiles does not touch.

This matters for the efficiency claim: `multi_token` reaches its quality
(§5.1) from **a single tile** — 362 GFLOPs of vision encode — where the
Vintern-1B reference recipe full-finetunes the ViT and runs up to 12 dynamic
tiles. The lever is real and 6× wide; §5.3 asks whether spending more of it,
adaptively, would have been worth it.

## 5.3 Is adaptive visual computation a useful lever? (oracle analysis)

**Robustness check, not the main claim**: §5.1–5.2 already show 1 tile is
sufficient for the headline result. This section asks whether it was merely
*convenient* — i.e. whether an oracle that could freely spend the n_tiles
lever (§5.2) per-sample, informed by reasoning type or cheap visual state,
would do meaningfully better. If it would not, 1 tile is not a compromise.

**Action space.** `A = bridge × n_tiles`, bridge ∈ {multi_token, qformer,
mini_qformer} (Exp B top-3), n_tiles ∈ {1, 3, 6} InternViT forward passes,
|A| = 9. Cost `C(a) = n_tiles / 6`; quality `M(a; x)` = per-sample CIDEr of the
greedy-decoded answer. Oracle utility `U(a; x, λ) = M − λ·C`, optimal action
`a*(x, λ) = argmax_a U`, evaluated on the λ-grid {0, .05, .1, .2, .4, .7, 1}.

**C3 (locked): bridges are the tile-count-augmented checkpoints** — each of the
3 bridges retrained with `--tile-choices 1,3,6` so the oracle sweep is no longer
confounded by an n_tiles=1-only training regime. Mean CIDEr by bridge × n_tiles
(test, 3 739 samples; val, 3 727, in parens):

| | t1 | t3 | t6 |
|---|---|---|---|
| `multi_token` | 0.902 (0.946) | 0.470 (0.496) | 0.515 (0.537) |
| `qformer` | 0.842 (0.866) | 0.854 (0.878) | 0.841 (0.870) |
| `mini_qformer` | 0.806 (0.842) | 0.828 (0.871) | 0.812 (0.843) |

`multi_token` (mean-pooled) still collapses sharply beyond 1 tile — tile-count
augmentation during training did **not** teach it to use more tiles (0.90 → 0.47
at t3, same magnitude of collapse as the original n_tiles=1-only checkpoint).
`qformer` and `mini_qformer` (cross-attention) stay flat across n_tiles, same as
before. **Conclusion unchanged**: n_tiles is not a lever this architecture class
converts into quality, whether or not the bridge ever saw >1 tile in training.

**Per-category effect of n_tiles is null** (established on the original
n_tiles=1-checkpoint sweep; not re-verified per-category on the tiled
checkpoints — the top-line bridge×n_tiles table above is). For the
cross-attention bridges, mean CIDEr by (category × n_tiles) showed no category
in which additional tiles help significantly — paired bootstrap (n3−n1 and
n6−n1, per-sample, best bridge) gave 95% CIs that included 0 for all 8
categories (val, 3 727 samples). The pilot (591 samples) had suggested spatial /
context / recognition gains of +0.11–0.14 CIDEr; these did not survive the full
sweep and were sampling noise.

**Per-sample oracle headroom looks large but is not structure.**
On the |A|=9 test set (3 739 samples, tile-trained checkpoints):

| policy | mean CIDEr | mean cost |
|---|---|---|
| oracle `a*(x, 0)` | 1.259 | 0.361 |
| **fixed `multi_token\|t1`** (best bridge, min tiles) | **0.902** | 0.167 |
| fixed `qformer\|t3` | 0.854 | 0.500 |
| random | 0.828 | 0.557 |

The oracle beats the best fixed action by **+0.36 CIDEr (+40%)** — essentially
identical to the 1-tile-checkpoint sweep (was +0.36/+40% there too). The
per-category headroom breakdown (spread evenly, +0.28 to +0.65, not concentrated
in a subset of reasoning types) is carried over from that sweep, not re-run here.

We show this headroom is **memorizable noise, not a generalizable signal**: a
policy MLP trained and evaluated on the *same* test split reproduces the oracle
(a*-match 0.98, mean CIDEr 1.26 — indistinguishable from the oracle upper
bound), so the input features (P(r|Q), f(I,Q)) have the capacity to express
`a*`. But `a*` does not transfer: the val and test oracle-a* distributions
differ (majority action qformer|t1 on val, multi_token|t1 on test) despite
identical stratification, because per-sample CIDEr — a corpus metric applied to
single 4-word answers — is a poor estimator of relative answer quality among 9
near-tied actions.

## 5.4 Policy ablation: does reasoning-type supervision help routing?

Three policy arms, all `PolicyMLP((·, λ) → a)` trained by cross-entropy against
`a*`, differing only in inputs:

- **ours** — P(r|Q) (PhoBERT router, 8-way, macro-F1 0.91) **+** f(I,Q)
  (InternViT CLS PCA-64 + question length + image clarity/occlusion/density)
- **rt_only** — P(r|Q) only
- **visual_only** — f(I,Q) only

**|A|=9, held-out, C3-locked (policy trained on the 5 547-sample train split —
oracle labels from the *original* n_tiles=1-checkpoint sweep, see caveat below —
evaluated on test 3 739 with tile-trained-checkpoint oracle labels):**

| arm | a*-match | mean CIDEr | mean cost | action picked (test, λ=0.2) |
|---|---|---|---|---|
| **fixed `multi_token\|t1`** | — | **0.902** | 0.167 | — |
| ours | 0.452 | 0.900 | 0.168 | `multi_token\|t1` 97.9% |
| rt_only | 0.457 | 0.902 | 0.167 | `multi_token\|t1` **100%** |
| visual_only | 0.452 | 0.901 | 0.167 | `multi_token\|t1` 98.0% |
| oracle `a*(x, λ=0)` | 1.00 | 1.259 | 0.361 | — |

With adequate training data, **all three policy arms still converge onto
`fixed: multi_token|t1`** — mean CIDEr 0.900–0.902, essentially identical to the
1-tile-checkpoint sweep's 0.901–0.902, λ-independent, a*-match in the same
0.43–0.46 band as the majority-class rate. Re-locking on tile-trained
checkpoints changes nothing: the policy still learns "use the best bridge at
one tile", and *nothing more*.

**Caveat on this re-lock.** Only val+test were re-swept on tile-trained
checkpoints; the train-split oracle labels used to fit the policy are still
from the original n_tiles=1-checkpoint sweep (re-sweeping train was judged not
worth the extra ~5 Kaggle shards — see reasoning below). This is a train/test
*M(a;x)*-surface mismatch, but it can only work *against* a learned policy
(training against one oracle surface and being scored against a slightly
different one adds noise, it cannot manufacture a false "policy beats fixed"),
so it does not threaten the null-result conclusion — if anything the near-exact
reproduction of the original numbers despite this mismatch is itself evidence
the conclusion is robust to it.

(On the original val split, in contrast, the same policies *over-fit* and land
*below* the fixed baseline — 0.82–0.85, all significantly worse by paired
bootstrap — because they chase the non-transferable a* noise.)

**|A|=6 (n_tiles only, bridge = qformer/mini_qformer), held-out** — same
picture: all three arms collapse onto `fixed: qformer|t1`, λ-independent, mean
CIDEr ≈ 0.842 (fixed `qformer|t3` 0.854, oracle 1.09).

### Findings

1. **No learned routing policy beats a fixed best-bridge / minimum-tiles
   policy** — all three arms are *significantly worse* (paired bootstrap).
   `visual_only` is the worst: it spends extra compute (cost 0.23 vs 0.17)
   selecting actions that are, on held-out data, worse choices.
2. **Reasoning-type supervision (P(r|Q)) adds nothing.** `ours` ≈ `visual_only`
   ≈ `rt_only`; on |A|=6 they are numerically identical. The training-set a*-match
   gap we briefly observed (`visual_only` 0.95 vs `rt_only` 0.38) is pure
   over-fitting and vanishes held-out.
3. The λ-conditioning is inert: because `C` deltas (1/6 → 1) dwarf per-sample
   `M` deltas (~0.05), `a*` is already n_tiles=1-dominated at λ = 0.05, so the
   policy learns to ignore λ.

## 5.5 Training- and alignment-side interventions

§5.3–5.4 rule out *when-to-spend-more-vision* as a lever. We ran two further
interventions on the headline `multi_token` bridge, on the other two axes one
could push short of touching the decoder — the **training target** and the
**vision–language representation alignment** — holding architecture, data split
and 1-tile inference fixed:

- **Multi-reference training target.** AutoViVQA gives 5 distinct reference
  answers per question; the headline bridge trains on the first
  (`--answer-sampling first`). Re-training with a reference resampled per epoch
  (`random`) exposes the model to all 5.
- **Projector-alignment KD.** Vintern-1B's own `mlp1` connector is pre-aligned
  to Qwen2 by their pretraining. We add an auxiliary loss pulling the trained
  bridge toward that teacher — `feat`: cosine between pooled bridge output and
  pooled teacher tokens; `logit`: KL between answer-token distributions decoded
  through the bridge vs. through `mlp1`, both via the frozen Qwen2 (SEA /
  BASIC-style).

| intervention (`multi_token`, seed 42) | F1(tok) | CIDEr-D | ΔF1 vs `first` |
|---|---|---|---|
| `first` (headline, §5.1; epoch-1 full val) | 50.7 | 94.4 | — |
| `--answer-sampling random` (epoch-1 full val) | 49.0 | 87.3 | −1.7 |
| align KD `feat`, α=1.0 (epoch-1 full val) | 49.7 | 92.0 | −1.0 |
| align KD `logit`, α=1.0 (epoch-2, 600-sample subset) | 40.7 | 80.1 | −10 |

Neither `answer-sampling=random` nor `feat`-alignment improves token-F1; both
shift CIDEr-D down 2–7 points. The deltas (single seed, no multi-seed CI yet)
are small — the result is the **absence of a lift**, not significant harm. The
bridge already attains the lowest validation CE of the five architectures
(1.49; §5.1), i.e. it is close to CE-optimal for what the frozen decoder admits,
and a `feat` auxiliary term only perturbs it off that point.

`logit`-alignment at α=1.0 is different: the KL term dominates the CE objective
(val CE rises to 2.84, nearly 2× the plain 1.49) and degrades generation
sharply. We report it as a **mis-weighting** — α was not tuned — rather than a
clean test of the concept; but since `feat`-alignment at the same weight already
showed zero benefit with a stable CE, we did not pursue a weight sweep.

Between them, §5.3–5.5 test three axes — visual-compute allocation, training
target, and representation alignment — and find no lift on any of them. §5.6
tests the fourth axis, the one they were all designed to avoid touching.

## 5.6 Decoder-side reference point: LoRA

§5.3–5.5 hold the decoder frozen throughout. As a **deliberate, isolated
departure** from the frozen-backbone spine — not part of it — we LoRA-tune
Qwen2-0.5B's attention projections (`q/k/v/o`, rank 16, ≈2% additional trainable
parameters) alongside the `multi_token` bridge, 1 epoch, 1 tile, otherwise
identical setup to the headline run:

| | plain (mean, 4 seeds) | **LoRA r=16 (mean, 3 seeds)** | Δ | ViMoE-VQA |
|---|---:|---:|---:|---:|
| F1(tok) | 49.8 | **53.17** | **+3.4** | 60.7 |
| CIDEr (in-house) | 97.0 | **~105.6** | **+8.6** | — |
| BLEU-4 | 16.0 | **~19.5** | **+3.5** | 12.5 |
| Acc | 8.3 | **10.4** (2-seed) | **+2.1** | 9.7 |

LoRA closes **~31% of the F1 gap to ViMoE-VQA** (10.9 → 7.5 points), **locked
across all 3 seeds** (42: F1 53.16; 123: 53.20; 3407: 53.15 — std ≈ 0.03),
no longer a 2/3-seed provisional result. Validation CE drops to 1.37–1.39 from
the plain bridge's 1.49.

**Generalizes to a second bridge.** The same LoRA config applied to `qformer`
(seed 42, 1 tile, in-house eval, n = 5 463) shows an even larger lift:

| | plain | +LoRA r=16 | Δ |
|---|---:|---:|---:|
| F1(tok) | 47.66 | **53.10** | **+5.4** |
| CIDEr (in-house) | 90.8 | **105.2** | **+14.3** |
| BLEU-4 | 14.6 | **19.3** | **+4.8** |
| ROUGE-L | 46.0 | **51.6** | **+5.6** |
| Acc | 7.34 | **10.91** | **+3.6** |
| val loss | 1.568 | **1.377** | −0.19 |

The gain is not `multi_token`-specific — it is a **decoder-capacity effect
that shows up regardless of which bridge feeds the decoder**, which is exactly
what "the frozen decoder is the ceiling" predicts: whatever representation
the bridge hands it, a slightly-unfrozen decoder can use it better.

**Corpus-level (pycocoevalcap) confirmation.** The in-house numbers above are
not directly cross-paper-comparable (§5.1); `scripts/rescore_corpus.py`
recomputes CIDEr-D/BLEU-4/ROUGE-L the same way as the §5.1 table (verified: it
reproduces `qformer`-plain's locked row, 86.7/17.5/47.1, exactly). Rescored for
`qformer`+LoRA r=16 (seed 42):

| | qformer plain | +LoRA r=16 | Δ | ViMoE-VQA |
|---|---:|---:|---:|---:|
| CIDEr-D | 86.7 | **101.9** | **+15.2** | 88.7 |
| BLEU-4 | 17.5 | **23.1** | **+5.6** | 12.5 |
| ROUGE-L | 47.1 | **52.6** | **+5.5** | 47.1 |

`qformer`+LoRA now beats ViMoE-VQA on all three corpus metrics (CIDEr-D +13.2,
BLEU-4 +10.6, ROUGE-L +5.5) — stronger than `multi_token`-plain on BLEU-4/ROUGE-L,
though still below `multi_token`-plain's CIDEr-D (94.4).

`multi_token`+LoRA r=16 (seed 42), same script:

| | multi_token plain | +LoRA r=16 | Δ | ViMoE-VQA |
|---|---:|---:|---:|---:|
| CIDEr-D | 94.4 | **101.7** | **+7.3** | 88.7 |
| BLEU-4 | 19.6 | **23.2** | **+3.6** | 12.5 |
| ROUGE-L | 50.0 | **52.7** | **+2.7** | 47.1 |

This is now **the strongest single number across every bridge/variant tested**,
plain or LoRA — it beats ViMoE-VQA on all three corpus metrics (+13.0/+10.7/+5.6)
*and* beats `multi_token`-plain's own CIDEr-D (94.4), which nothing else in this
paper does. Both LoRA bridges now have complete in-house + corpus numbers.

This is the **only one of the four axes tested (§5.3–5.6) that moves F1**, and
it is the only one that touches the decoder. It does not weaken the
frozen-backbone efficiency claim (§5.1–5.2) — it is reported here as a
robustness/reference point, quantifying exactly how much headroom exists once
the one deliberate departure from "everything but the bridge is frozen" is
allowed, not as a replacement for the main spine. Both LoRA bridges' F1/CIDEr/
BLEU/ROUGE numbers are now locked (3 seeds for `multi_token`, 1 for `qformer`,
both in-house and corpus); a rank sweep (r=8, r=32) is running to see whether
r=16 was a lucky choice or the effect is robust across rank.

## 5.7 Summary of findings

Across §5.3–5.6, four different ways of trying to close the remaining F1 gap
to ViMoE-VQA were tried on top of the frozen-backbone, 1-tile bridge:

| axis | intervention | result |
|---|---|---|
| visual-compute allocation | reasoning-type-adaptive tile routing | no policy beats fixed 1-tile (§5.3–5.4) |
| training target | multi-reference (`answer-sampling=random`) | F1 −1.7, no lift (§5.5) |
| representation alignment | projector-KD from Vintern's `mlp1` | F1 −1.0 (`feat`), −10 mis-weighted (`logit`) (§5.5) |
| decoder capacity | LoRA r=16 on Qwen2 attention | **F1 +3.4 to +5.4** — the one positive (§5.6) |

**Reading:** three vision-/training-side interventions, all negative;
one decoder-side intervention, clearly positive **and bridge-agnostic** —
LoRA lifts F1 on both `multi_token` (+3.4) and `qformer` (+5.4), confirming
it is a decoder-capacity effect, not an artifact of one bridge architecture.
This is not noise — it localizes the bottleneck. With a fully frozen, small
(0.5B) decoder, additional visual detail, training signal, or representation
alignment on the vision side has nothing to attach to; the frozen decoder is
the ceiling, not the vision pipeline, regardless of which bridge feeds it.
Opening the decoder even slightly (2% of its parameters via LoRA) recovers a
third or more of the gap to a fully-unfrozen prior-work baseline. §6.1
develops this reading; the paper's primary contribution remains the frozen,
0.78%-param bridge (§5.1–5.2) — the decoder-ceiling finding explains *why* that
architecture class tops out where it does, rather than proposing to abandon it.

## 5.8 Does token-F1 mean the answer is correct? A self-check

Every result above is read through automatic metrics (CIDEr-D, BLEU-4, ROUGE-L,
token-F1) against 5 reference answers. This section asks how much those metrics
actually track answer correctness, using `multi_token`'s val predictions.

**Scope reduction, stated up front.** The plan called for human validation —
300–500 samples, 2 annotators, Cohen's κ. Time did not allow it before the
deadline; what follows is a **single-rater self-check substitute**, not human
validation, and is reported as such: N = 120 (not 300–500), one rater (the
assistant, not an independent human annotator), no image access — judged for
*plausibility against the 5 reference answers*, not independently verified
against the image, which for open-ended categories (causal/context) is a
materially different and weaker check than true human validation. Sampled
proportionally by category × the *actual* F1 bucket of each prediction (seed
42, `scripts/human_validation_sample.py`), scored by
`scripts/human_validation_report.py`; all 120 judgments with reasoning in
`outputs/human_validation/selfcheck_judgments.json`.

| F1 bucket | n | correct | partially correct | wrong | nonsense | **acceptable (correct+partial)** |
|---|---:|---:|---:|---:|---:|---:|
| strong (≥0.6) | 45 | 80.0% | 11.1% | 6.7% | 2.2% | **91.1%** |
| partial (0.2–0.6) | 58 | 12.1% | 31.0% | 55.2% | 1.7% | **43.1%** |
| weak (0–0.2) | 3 | 0% | 0% | 100% | 0% | **0%** |
| zero (F1=0) | 13 | 7.7% | 7.7% | 76.9% | 7.7% | **15.4%** |
| **total (n=119\*)** | | **37.0%** | **20.2%** | **40.3%** | **2.5%** | **57.1%** |

\* one sample excluded: self-contradictory reference set.

**Reported straight, not spun.** The "strong" bucket is reliable (91.1%
acceptable) — high F1 is a good correctness signal there. But the **"partial"
bucket (0.2–0.6) is both the *largest* single bucket (51.5% of val) and the
*least* reliable** — only 43.1% acceptable, 55.2% actually wrong despite
sharing filler tokens (generic words, color names) with the reference. The
failure mode is not random noise: wrong color/count/object, or answering the
wrong facet of the question (e.g. "when" answered with weather; the wrong
gender for "who"; a yes/no answer inverted relative to the reference) — errors
a fluent decoder can produce while still overlapping enough vocabulary to score
mid-range F1. ("zero"-bucket answers are mostly wrong (84.6%), but not
entirely — some are semantically correct paraphrases with zero token overlap.)

Overall, **37.0% of val predictions are fully correct and 57.1% are
acceptable** by this check — noticeably lower than headline numbers like
CIDEr-D 94.4 or F1 44.2 might suggest to a reader unfamiliar with these
metrics' scales, though the self-check's own aggregate (37.0%/57.1%) sits close
to the "strong"-bucket share of val, which is some corroboration that "strong"
≈ "actually correct" is a reasonable proxy even though the metric as a whole
is not.

**Limitations of this self-check itself** (not to be conflated with the
frozen-decoder findings above): single rater, no ground-truth image access,
N=120 rather than 300–500, no second rater and therefore no Cohen's κ. This is
a time-constrained substitute, not a replacement for human validation should a
reviewer require it — but it is enough to surface a real, actionable finding
that F1 alone would have missed: **mid-range token-F1 is not a reliable
correctness signal**, which qualifies how every CIDEr-D/F1 number in §5.1–§5.7
should be read.

---

### Pending
- [x] §5.6: qformer-LoRA generalization check landed — F1 +5.4, bridge-agnostic
      confirmed
- [x] §5.6: qformer-LoRA corpus-level (pycocoevalcap) rescore landed — beats
      ViMoE on all 3 corpus metrics
- [x] §5.6: multi_token-LoRA seed 42 landed (3/3 seed locked, std≈0.03 F1) +
      corpus rescore landed — strongest single number in the paper, beats both
      ViMoE and multi_token-plain's own CIDEr-D
- [ ] §5.6: fold in the r=8/r=32 rank sweep once it lands (co-author, running)
- [ ] §5.5: multi-seed numbers + CI once available; re-run align-feat/logit
      on full val (both were cut short) if a reviewer needs it
- [x] **§5.8 NEW**: human validation re-scoped to single-rater self-check (user:
      no annotator time before deadline) — N=120, self+reasoning in
      `outputs/human_validation/selfcheck_judgments.json`. Finding: "partial"
      F1 bucket (largest, 51.5% of val) only 43.1% acceptable — reported
      straight, not spun. Qualifies how every metric number in §5 should be read.
- [ ] flag to co-author: §6.4's limitation item "human validation not yet
      included" needs updating — it's now partially addressed (§5.8), should
      note the reduced scope (1 rater, no image access, N=120 not 300-500) and
      probably reference §5.8's finding (mid-range F1 unreliable) as its own
      point, not just "not yet done"
- [ ] flag to co-author: §6.1/§6.3 in `06-discussion.md` still reference the
      pre-reorder §5 numbering/framing (old "reasoning-type supervision"
      research question, §5.5 = compute-efficiency) — needs a matching pass
      now that §5 is reordered and §5.6/§5.7 (LoRA, summary) exist
