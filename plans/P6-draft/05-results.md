# §5 Experiments and Results

> Draft. Numbers marked **[PRELIM]** may shift after (a) re-training the policy on
> the |A|=9 *train* split (5 547 samples vs the current 3 727 *val* samples) and
> (b) the tile-augmented `multi_token` retrain (running). Core findings are stable.

## 5.1 Bridge architecture comparison

(→ owned by co-author; see `plans/results-grouped-split.md`. One-line summary
for cross-reference: on the grouped 70/15/15 split, `multi_token` is the
strongest bridge — corpus CIDEr-D **0.94 val / 0.90 test**, above the reported
ViMoE/Tuong-MoE 0.887 and GPT-5 0.842 — while trailing on word-level F1.)

## 5.2 Is visual computation a useful lever? (oracle analysis)

**Action space.** `A = bridge × n_tiles`, bridge ∈ {multi_token, qformer,
mini_qformer} (Exp B top-3), n_tiles ∈ {1, 3, 6} InternViT forward passes,
|A| = 9. Cost `C(a) = n_tiles / 6`; quality `M(a; x)` = per-sample CIDEr of the
greedy-decoded answer. Oracle utility `U(a; x, λ) = M − λ·C`, optimal action
`a*(x, λ) = argmax_a U`, evaluated on the λ-grid {0, .05, .1, .2, .4, .7, 1}.

Bridge checkpoints are the n_tiles=1-trained models from §5.1; `multi_token` is
mean-pooled and, as expected, degrades sharply when fed >1 tile
(CIDEr 0.90 → 0.47 at n_tiles=3), so the oracle never selects `multi_token|t3`
or `|t6`. `qformer` and `mini_qformer` (cross-attention) are stable across
n_tiles.

**Per-category effect of n_tiles is null.** For the cross-attention bridges,
mean CIDEr by (category × n_tiles) shows no category in which additional tiles
help significantly — paired bootstrap (n3−n1 and n6−n1, per-sample, best bridge)
gives 95% CIs that include 0 for all 8 categories (val, 3 727 samples). The
pilot (591 samples) suggested spatial / context / recognition gains of
+0.11–0.14 CIDEr; these did not survive the full sweep and were sampling noise.

**Per-sample oracle headroom looks large but is not structure.**
On the |A|=9 test set (3 739 samples):

| policy | mean CIDEr | mean cost |
|---|---|---|
| oracle `a*(x, 0)` | 1.26 | 0.36 |
| **fixed `multi_token\|t1`** (best bridge, min tiles) | **0.90** | 0.17 |
| fixed `qformer\|t3` | 0.85 | 0.50 |
| random | 0.77 | 0.56 |

The oracle beats the best fixed action by **+0.36 CIDEr (+40%)**, and the gap is
spread evenly across all 8 categories (per-category headroom +0.28 to +0.65) —
i.e. *not* concentrated in a subset of reasoning types.

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

**|A|=9, held-out (train on val 3 727 → eval on test 3 739)** [PRELIM]:

| arm | a*-match | mean CIDEr | mean cost | Δ vs fixed `mt\|t1` (paired bootstrap) |
|---|---|---|---|---|
| fixed `multi_token\|t1` | — | **0.902** | 0.167 | — |
| ours | 0.25 | 0.851 | 0.177 | **−0.055** [−0.074, −0.037] — significantly worse |
| rt_only | 0.18 | 0.845 | 0.167 | **−0.060** [−0.082, −0.039] — significantly worse |
| visual_only | 0.22 | 0.820 | 0.225 | **−0.100** [−0.122, −0.079] — significantly worse |
| majority-class a* | 0.43 | — | — | — |

**|A|=6 (n_tiles only, bridge = qformer/mini_qformer), held-out** — same
picture, all three arms collapse to *always `qformer|t1`*, λ-independent, mean
CIDEr ≈ 0.842 (vs fixed `qformer|t3` 0.854, oracle 1.09).

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

n_tiles is a genuine compute lever (InternViT GFLOPs ×6, P100 latency ×4 from
n_tiles=1→6). The negative routing result is therefore *not* "the lever is too
small to matter" — the lever is real, but the frozen 0.5 B language model cannot
convert the extra visual detail into better answers, in any reasoning category.

---

### Pending
- [ ] re-run 5.3 policies on the |A|=9 *train* split (5 547) — lock numbers
- [ ] tile-augmented `multi_token` oracle sweep → redo 5.2 headroom analysis if
      the pooled bridge trained *with* tiles exploits them
- [ ] human validation of 300–500 answers (2 raters, Cohen's κ) — §5.1/§6
