# §5 Experiments and Results

> Draft. Numbers marked **[PRELIM]** may shift after (a) re-training the policy on
> the |A|=9 *train* split (5 547 samples vs the current 3 727 *val* samples) and
> (b) the tile-augmented `multi_token` retrain (running). Core findings are stable.

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

**|A|=9, held-out (policy trained on the 5 547-sample train split, evaluated on
test 3 739):**

| arm | a*-match | mean CIDEr | mean cost | action picked (test, λ=0.2) |
|---|---|---|---|---|
| **fixed `multi_token\|t1`** | — | **0.902** | 0.167 | — |
| ours | 0.434 | 0.901 | 0.168 | `multi_token\|t1` 94.5% |
| rt_only | 0.442 | 0.902 | 0.167 | `multi_token\|t1` **100%** |
| visual_only | 0.437 | 0.901 | 0.168 | `multi_token\|t1` 97.3% |
| majority-class a* | 0.433 | — | — | — |
| oracle `a*(x, λ)` | 1.00 | 1.25 | 0.29 | — |

With adequate training data, **all three policy arms converge exactly onto
`fixed: multi_token|t1`** — mean CIDEr 0.901–0.902, identical to the fixed
baseline (0.902), λ-independent, a*-match ≈ the majority-class rate (0.43). The
policy correctly learns "use the best bridge at one tile", and *nothing more*.
(Trained on the smaller 3 727-sample val split instead, the same policies
*over-fit* and land *below* the fixed baseline — 0.82–0.85, all significantly
worse by paired bootstrap — because they chase the non-transferable a* noise.)

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

---

### Pending
- [x] re-run 5.3 policies on the |A|=9 *train* split (5 547) — **done, locked**:
      all arms → `fixed: multi_token|t1` (0.901–0.902), a*-match ≈ majority 0.43
- [x] 5.5 compute-efficiency table added (P1 v8 profiling)
- [ ] **PIVOT (co-author, 2026-09-03): efficiency is now the primary story**,
      routing/oracle result demoted to a supporting ablation ("1 tile is not a
      compromise"), P(r|Q) reasoning-type demoted to a small ablation. §5 content
      stays valid; needs section reorder + reframing once numbers re-lock.
- [ ] **re-lock 5.2/5.3 on tile-CAPABLE checkpoints.** Current oracle uses the
      n_tiles=1-trained bridges — the confound a reviewer hits. Tiled retrains
      launched: `multi_token`→acc11, `qformer`→acc9, `mini_qformer`→acc10
      (~2026-09-04). When they land: re-run oracle sweep (val+test) + policy
      ladder on the tiled checkpoints, re-lock the tables. Expected: same
      conclusion (no policy beats fixed), cleaner numbers.
- [ ] tile-augmented `multi_token` oracle sweep (retrain on acc11) →
      redo 5.2 headroom analysis if the pooled bridge trained *with* tiles
      exploits them; if it collapses like the n_tiles=1 checkpoint, no change
- [ ] human validation of 300–500 answers (2 raters, Cohen's κ) — §5.1/§6
- [ ] `--answer-sampling random` multi_token run (co-author) — F1 gap chase,
      §5.1 only, does not touch the oracle
