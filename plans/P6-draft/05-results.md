# §5 Experiments and Results

> Draft. §5.2/§5.3 are **LOCKED on tile-trained checkpoints** (C3, 2026-09-05) —
> the oracle/policy numbers below are re-swept on bridges retrained *with*
> tile-count augmentation (`--tile-choices 1,3,6`), closing the reviewer confound
> of evaluating an n_tiles-adaptive oracle against n_tiles=1-only-trained models.
> Result: **no change** — the tile-trained checkpoints reproduce the same
> collapse/flatness pattern and the same policy-converges-to-fixed conclusion as
> the original 1-tile-checkpoint sweep, within noise. One open caveat: the §5.3
> policy is trained on the *original* (1-tile-checkpoint) train-split oracle
> labels — train was not re-swept (see §5.3 note) — while it's evaluated against
> the tiled-checkpoint test labels.

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

## 5.2 Is visual computation a useful lever? (oracle analysis)

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

## 5.3 Policy ablation: does reasoning-type supervision help routing?

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

## 5.4 Answer to the research question

> *Does explicit reasoning-type supervision improve the allocation of visual
> computation beyond model-internal signals?*

**No — and, for this lightweight VLM class, neither do model-internal signals.**
The per-sample optimal action carries a large apparent headroom (+40% CIDEr) but
it is not a learnable function of question type or cheap visual state; it is an
artifact of CIDEr's per-example variance. Adaptive visual-compute allocation
does not beat a well-chosen fixed policy (best bridge at minimum tiles), and
reasoning-type labels — which on this auto-generated dataset largely encode
question surface form (§4, router F1 0.91) — contribute nothing on top.

## 5.5 Compute–efficiency (P1 profiling)

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
tiles. The negative routing result (§5.2–5.4) is therefore *not* "the lever is
too small to matter": the lever is real and 6× wide, but the frozen 0.5B
language model cannot convert the extra visual detail into better answers, in
any reasoning category — so the cheapest point on the lever is also the best.

## 5.6 Training- and alignment-side interventions

The routing result (§5.2–5.4) rules out *when-to-spend-more-vision* as a lever.
We ran two further interventions on the headline `multi_token` bridge, on the
other two axes one could push — the **training target** and the
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
showed zero benefit with a stable CE, we did not pursue a weight sweep. We read
the convergence of the three negatives in §6.1.

---

### Pending
- [x] §5.6 align-KD `logit` row filled (α=1.0, KL dominates → val CE 2.84, F1 40.7)
- [x] 5.5 compute-efficiency table added (P1 v8 profiling)
- [x] re-run 5.3 policies on the |A|=9 *train* split (5 547 samples, original
      n_tiles=1-checkpoint oracle) — locked: all arms → `fixed: multi_token|t1`
      (0.901–0.902), a*-match ≈ majority 0.43
- [x] **C3 DONE (2026-09-05): §5.2/§5.3 re-locked on tile-trained checkpoints.**
      All 3 bridges retrained with `--tile-choices 1,3,6`; oracle re-swept val+test
      (`outputs/oracle_{val,test}_tiled/`, `scripts/analyze_A9_tiled.py`). Result:
      no change — same collapse/flatness pattern, same policy-converges-to-fixed
      conclusion, numbers within noise of the original sweep. Confound closed.
      Known limitation (stated in §5.3): train-split oracle labels used to fit the
      policy are still from the original (non-tiled) sweep — biases against, not
      for, a learned policy beating fixed.
- [ ] **PIVOT (co-author, 2026-09-03): efficiency is now the primary story.**
      Routing/oracle result (§5.2–5.4) demoted to a supporting ablation ("1 tile is
      not a compromise"), P(r|Q) reasoning-type demoted to a small ablation.
      Content is now fully locked (incl. C3) — still needs the section reorder +
      reframing pass toward efficiency-primary.
- [ ] §5.6: swap in multi-seed numbers + CI once C2 lands; re-run align-feat/logit
      on full val (both were cut short) if a reviewer needs it
- [ ] human validation of 300–500 answers (2 raters, Cohen's κ) — §5.1/§6
