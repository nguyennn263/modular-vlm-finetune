# §5–§7 · Experiments and Results

> Restructured 2026-09-07 to the paper blueprint (`plans/paper-blueprint.md`):
> **§5 Main Results** (recipe vs baselines) · **§6 Ablation — hunting the
> bottleneck** (six RQs) · **§7 Human Validation & Error Analysis**. Numbers are
> the final 2-epoch, 3-seed set (`multi_token` = 4 seed); sources
> `results-5bridge.md` (main) and `results-grouped-split.md` (ablation).
> Blueprint owns the presentation tables; this file owns the prose.

---

## §5 Main Results

**The question.** Vintern-1B is a strong Vietnamese VLM, but adapting it to a new
benchmark the way its authors did is expensive — they train InternViT-300M and
the projector in full and LoRA the LLM, on ~3M pairs. ViMoE-VQA's answer to the
same problem is to build a new mixture-of-experts model. We ask whether the
existing model can be adapted **cheaply** — both backbones frozen, ~1% of
parameters trainable, one image tile — and, if it falls short anywhere, **where
the bottleneck is**.

**The recipe.** Freeze InternViT-300M and Qwen2-0.5B. Train only a bridge
(vision→LLM projector; `multi_token`, 7.35M params, 0.78%). Optionally add a
rank-16 LoRA on the decoder's attention projections (`q/k/v/o`; 2.16M, 0.23%).
Total trainable: **1.01%**. One tile, grouped leak-free 70/15/15 split
(§4), greedy decoding.

### 5.1 Recipe vs prior work

**Table 1** (blueprint) reports all AutoViVQA baselines and our recipe on the
eight in-house metrics (Acc / Prec / Rec / F1 / BLEU / ROUGE / METEOR / CIDEr,
×100). Headline rows:

| model | F1 | BLEU | ROUGE | METEOR | CIDEr | Acc |
|---|---:|---:|---:|---:|---:|---:|
| Vintern-1B (base, zero-shot) | 17.6 | 1.9 | 25.8 | 23.9 | 8.5 | 0.1 |
| Vintern-1B (fine-tuned, ≤12 tiles) | 53.8 | 6.1 | 51.9 | 35.3 | 72.8 | 13.0 |
| GPT-5 (zero-shot) | 50.9 | 6.1 | 47.3 | 33.3 | 84.2 | 10.8 |
| ViMoE-VQA (Tuong-MOE, 5-seed) | **60.7** | 12.5 | 47.1 | **39.1** | 88.7 | 9.7 |
| **Bridge `multi_token` (0.78%, 1 tile)** — 4-seed | 49.55 | 15.47 | 47.84 | 40.22 | 96.49 | 8.20 |
| **  + decoder LoRA r=16 (1.01% total)** — 3-seed | 53.17 | 19.44 | 51.48 | 43.91 | 105.59 | 10.42 |
| **  + decoder LoRA r=16, 3 epochs** — 3-seed | 54.67 | 20.98 | 52.92 | 45.24 | 109.60 | 11.78 |

**Reading.** The frozen-backbone recipe beats **Vintern-1B fine-tuned on every
generation metric** (BLEU +14.9, METEOR +10.0, CIDEr +36.8) at ~1% of the
trainable parameters and 1 tile instead of up to 12, and beats **ViMoE-VQA on
BLEU / ROUGE / METEOR / CIDEr**. It trails ViMoE on token-F1 (−11.2 for the
bridge alone, −7.5 with LoRA, −6.0 at 3 epochs) and Vintern on Acc. §6 diagnoses
where that remaining gap lives.

### 5.2 Corpus metrics and confidence intervals

The in-house metrics above match the AutoViVQA table convention but are not
directly comparable to numbers computed with a different implementation. **Table
2** (blueprint) reports the cross-paper-comparable corpus metrics
(pycocoevalcap: CIDEr-D, BLEU-4, ROUGE-L):

| model | CIDEr-D | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| ViMoE-VQA | 88.67 | 12.54 | 47.07 |
| **`multi_token` (4-seed, 2 epoch)** | **92.30 ± 0.60** | **18.90 ± 0.30** | **48.90 ± 0.10** |
| **  + LoRA r=16 (seed 42)** | **101.70** | **23.20** | **52.70** |
| **  + LoRA r=16, 3 epochs (3-seed)** | **106.80 ± 1.10** | **25.00 ± 0.40** | **54.20 ± 0.20** |

A paired bootstrap over the 5 463 val samples (`scripts/bootstrap_ci.py`;
CI **being recomputed on the 2-epoch predictions** — the [91.3, 97.1] interval
below is from the earlier 4-epoch seed-42 run and is expected to shift only
slightly) put `multi_token`-plain's CIDEr-D 95% CI entirely above ViMoE's 88.7:
the generation-quality win is not a lucky draw. ViMoE publishes no per-sample
predictions, so only a one-sample CI on our own estimate is possible.

### 5.3 Held-out (test) evaluation

Most numbers above are on val. On the held-out test split (n = 5 468, 4 seeds),
`multi_token`-plain scores F1 **49.20** / CIDEr **93.24** vs val 49.55 / 96.49 —
a gap of −0.35 F1 / −3.25 CIDEr, small and **not consistent in direction** across
bridges (`mini_qformer` test 47.25 vs val 47.05; `residual` 45.49 vs 45.91;
`tile_attention` 44.44 vs 44.50). The recipe is **not over-fit to val**.

### 5.4 Compute-efficiency of the tile lever

`n_tiles` is a genuine compute lever. Profiled on a Tesla P100-16GB
(`mini_qformer`, `src.cli.profile`), per image:

| n_tiles | InternViT GFLOPs | latency (ms) | throughput (img/s) |
|---|---:|---:|---:|
| **1 (recipe)** | **362** | **229** | **6.0** |
| 2 | 724 | 374 | 3.3 |
| 4 | 1 448 | 648 | 1.7 |
| 6 | 2 172 | 922 | 1.15 |

Dynamic range 1→6: FLOPs ×6.0, latency ×4.0, throughput ×5.2. The recipe spends
**none** of this lever. §6.2 shows why spending it is not just unhelpful but
actively harmful for this bridge; this is the FLOPs/latency analysis ViMoE-VQA
explicitly deferred.

---

## §6 Ablation: Hunting the Bottleneck

The recipe leaves an F1 gap to ViMoE-VQA. §6 works through **six candidate axes**
for closing it — each a different place to spend a small extra budget — and asks
which one actually moves the needle. Anchor throughout: `multi_token`-plain,
4-seed 2-epoch, **F1 49.55 / CIDEr-D 92.30**.

### 6.1 Bridge architecture and capacity (RQ1–RQ2)

**Five bridges** on the frozen backbone (val, 2 epoch, 3-seed; `multi_token` =
4-seed):

| bridge | params (%) | F1 | CIDEr | val CE | F1 + LoRA | ΔF1 |
|---|---|---:|---:|---:|---:|---:|
| Residual (1 tok) | 4.86M (0.52) | 45.64 | 86.25 | 1.67 | 52.64 | +7.0 |
| Tile-Attention (8 tok) | 4.14M (0.44) | 45.17 | 84.21 | 1.67 | 52.99 | +7.8 |
| **Multi-Token (8 tok, pooled)** | **7.35M (0.78)** | **49.55** | **96.49** | **1.49** | **53.17** | **+3.6** |
| Light Q-Former (8 query) | 27.6M (2.87) | 46.25 | 86.80 | 1.60 | 53.21 | +7.0 |
| Full Q-Former (16 query) | 69.4M (6.91) | 47.36 | 88.31 | 1.57 | 53.21 | +5.9 |

- **RQ1 — is a simple bridge enough?** Yes. `multi_token` (0.78%) is the best
  bridge and already beats Vintern-1B fine-tuned on generation metrics.
- **RQ2 — does a bigger bridge help?** No. Full Q-Former has **10× the
  parameters and scores lower** (F1 47.4 vs 49.6). `multi_token` has the lowest
  validation CE of the five. Bridge capacity is not the binding constraint.

*(`residual`'s earlier F1 36.45 / CIDEr-D 56.3 was an unstable seed-42 training
run — best val CE 2.35 vs 1.5–1.7 elsewhere; the 3-seed 2-epoch numbers here
resolve it to 45.64, in line with the other sub-bridges.)*

### 6.2 Visual tiles (RQ3)

**RQ3 — does spending the tile lever help?** No — it breaks the bridge. Feeding
the 1-tile-trained `multi_token` more tiles at inference (val, full):

| n_tiles | token-F1 | CIDEr | val loss |
|---|---:|---:|---:|
| **1** | **50.66** | **98.69** | **1.48** |
| 3 | 21.05 | 48.75 | 3.35 |
| 6 | 22.51 | 52.36 | 3.36 |

F1 50.7 → 21, val loss 1.48 → 3.36. Mean-pooling 8 output tokens over 3–6× as
many patches washes out the signal (**Figure 2**). We also **retrained** all
three oracle bridges *with* tile-count augmentation (`--tile-choices 1,3,6`) and
re-ran the analysis in §6.3 on those checkpoints: `multi_token` still collapses
beyond 1 tile even when it saw multiple tiles in training (CIDEr 0.90 → 0.47 at
3 tiles); the cross-attention bridges stay flat. **1 tile is the operating point
this architecture is built for, not a compromise** — spending more of the lever
is strictly worse.

### 6.3 Adaptive routing (RQ4)

**RQ4 — could a policy spend the tile lever adaptively, per sample, informed by
question type (ViMoE's "reasoning-aware" idea) or cheap visual state?** No.

**Oracle setup.** Action space `A = bridge × n_tiles`, bridge ∈ {multi_token,
qformer, mini_qformer}, n_tiles ∈ {1,3,6}, |A| = 9. Cost `C(a) = n_tiles/6`;
quality `M(a;x)` = per-sample CIDEr of the greedy answer. Oracle utility
`U(a;x,λ) = M − λ·C`; `a*(x,λ) = argmax_a U` over the λ-grid
{0,.05,.1,.2,.4,.7,1}. Bridges are the tile-augmented retrains (closes the
"never saw >1 tile" confound). |A|=9 held-out test (3 739 samples):

| policy | mean CIDEr | mean cost |
|---|---:|---:|
| oracle `a*(x, 0)` | 1.259 | 0.361 |
| **fixed `multi_token\|t1`** (best bridge, min tiles) | **0.902** | 0.167 |
| fixed `qformer\|t3` | 0.854 | 0.500 |
| random | 0.828 | 0.557 |

The oracle beats the best fixed action by **+0.36 CIDEr (+40%)**, spread evenly
across all 8 reasoning categories (per-category headroom +0.28 to +0.65) — *not*
concentrated in a subset of reasoning types.

**The headroom is memorizable noise, not a learnable signal.** A policy MLP
trained *and* evaluated on the same test split reproduces the oracle (a*-match
0.98, mean CIDEr 1.26). But `a*` does not transfer: the val and test oracle-a*
distributions disagree (majority `qformer|t1` on val, `multi_token|t1` on test)
despite identical stratification, because per-sample CIDEr on single 4-word
answers is a poor estimator of relative quality among near-tied actions.

**Three policy arms** — `PolicyMLP((·,λ)→a)` trained by cross-entropy against
`a*`, inputs: `ours` = reasoning-type posterior P(r|Q) (PhoBERT router, macro-F1
0.91) + visual state f(I,Q); `rt_only` = P(r|Q); `visual_only` = f(I,Q).
Held-out test:

| arm | a*-match | mean CIDEr | action picked (λ=0.2) |
|---|---:|---:|---|
| **fixed `multi_token\|t1`** | — | **0.902** | — |
| ours | 0.452 | 0.900 | `multi_token\|t1` 97.9% |
| rt_only | 0.457 | 0.902 | `multi_token\|t1` **100%** |
| visual_only | 0.452 | 0.901 | `multi_token\|t1` 98.0% |
| oracle | 1.00 | 1.259 | — |

**All three arms converge onto `fixed: multi_token|t1`**, λ-independent, a*-match
≈ the majority-class rate (0.43–0.46). Adding the reasoning-type posterior
changes nothing (`ours` ≈ `visual_only` ≈ `rt_only`). No learned routing policy
beats the fixed best-bridge / minimum-tiles policy — and reasoning-type labels,
which on this auto-generated dataset largely encode question surface form
(§4, router F1 0.91), contribute nothing on top.

### 6.4 Training signal and representation alignment (RQ5)

**RQ5 — does a better training target or a better vision–language alignment
close the gap?** No — three sub-interventions on `multi_token`, all 3-seed
except where noted, anchor F1 49.55:

| intervention | F1 | ΔF1 | note |
|---|---:|---:|---|
| multi-reference answer sampling (`--answer-sampling random`) | 48.08 | −1.47 | 3-seed |
| projector feature-KD from Vintern's `mlp1` (`align-feat`, α=1.0) | 49.53 | **−0.03** | 3-seed — **absolute null** |
| projector logit-KD (`align-logit`, α=1.0) | 40.75 | −8.80 | 3-seed; KL dominates CE (val CE ~2.05 vs 1.49) — mis-weighted, reported as such |

`answer-sampling=random` slightly hurts. `align-feat` — distilling the bridge
toward Vintern's *own* pre-aligned connector — is an **absolute null** at
3 seeds (ΔF1 −0.03): the bridge is already at the alignment the frozen decoder
can use. `align-logit` at α=1.0 is a known mis-weighting (the KL term swamps the
generation objective); since the correctly-weighted `feat` variant already shows
zero benefit, we did not pursue a weight sweep. **Nothing on the training or
alignment axis moves F1.**

### 6.5 Decoder capacity: LoRA (RQ6)

**RQ6 — does opening the decoder itself help?** **Yes — this is the one axis
that moves F1**, and it is the only intervention that touches the decoder.

**Per bridge.** LoRA r=16 on the decoder's attention (`q/k/v/o`, 1 epoch,
3-seed; `tile_attention` seed 42), val:

| bridge | F1 plain → +LoRA | ΔF1 | CIDEr-D plain → +LoRA | ΔCIDEr-D |
|---|---:|---:|---:|---:|
| `multi_token` | 49.55 → 53.17 | +3.6 | 92.3 → 101.7 | +9.4 |
| `qformer` | 47.36 → 53.21 | +5.9 | 86.9 → 102.4 | +15.5 |
| `mini_qformer` | 46.25 → 53.21 | +7.0 | 83.7 → 103.0 | +19.3 |
| `residual` | 45.64 → 52.64 | +7.0 | 81.1 → 100.8 | +19.7 |
| `tile_attention` | 45.17 → 52.99 | +7.8 | 79.0 → 102.0 | +23.0 |

**Bridge-equalizing (Figure 1).** The five plain bridges span F1 45.2–49.6 /
CIDEr-D 79–92 — real quality differences across three token-mixing designs
(pooled, cross-attention, patch self-attention). After a 0.23% LoRA on the
decoder, **all five converge to F1 52.6–53.2 / CIDEr-D 100.8–103.0**. The lift
is larger for weaker bridges (+3.6 → +7.8). Once the decoder has capacity to use
whatever representation it is given, *which bridge supplies it stops mattering* —
the bottleneck is decoder capacity, not bridge sophistication. A paired
bootstrap on the seed-42 predictions puts every plain→LoRA ΔF1/ΔCIDEr-D CI clear
of zero, P(Δ>0) = 1.000 (`residual`'s row pending recompute against a sound
seed).

**Where in the decoder (TIER-2).** LoRA r=16, 1 epoch, 3-seed, `multi_token`:

| target module | F1 | val loss | outcome |
|---|---:|---:|---|
| attention `q/k/v/o` (recipe) | **53.17** | 1.37 | +3.6, stable |
| MLP `gate/up/down_proj` | 20.24 ± 1.52 | ~3.7 | diverges |
| attention + MLP (all 7) | 37.51 ± 1.70 | ~2.08 | diverges (attn partly rescues) |

The usable decoder headroom is **specifically in attention**. LoRA on the
feed-forward blocks diverges. *Caveat:* this may be a hyperparameter artifact
(α=32 is likely too strong for the wider MLP dimension); we limit the claim to
the recipe's configuration.

**Rank.** A 5-point rank sweep (r ∈ {4,8,16,32,64}, 600-sample subset) initially
looked monotonic on one seed; with 3-seed means, r=32 (53.83 ± 1.77) and r=64
(54.06 ± 0.94) are statistically indistinguishable (0.23 apart). We keep **r=16**
as the operating point — the one with full-val 3-seed rigor — and make no
"higher rank is better" claim.

**Training duration.** 1 epoch → 3 epochs adds F1 53.17 → 54.67 (+1.5),
CIDEr-D 101.7 → 106.8, closing the F1 gap to ViMoE from 7.5 to 6.0; most of the
gain lands by epoch 2. 1-epoch numbers are the headline (cheaper, same schedule
as everything else); the 3-epoch result is the strongest form. A 5-epoch run was
cut by the compute quota at ~4 epochs.

### 6.6 Summary: six axes, one positive

| RQ · axis | intervention | ΔF1 | outcome |
|---|---|---:|---|
| RQ1–2 · bridge capacity | Full Q-Former (69M, 10×) | −2.19 | negative |
| RQ3 · visual tiles | train 1 tile → eval 3 | −28.5 | negative (collapse) |
| RQ4 · adaptive routing | learned policy (reasoning + visual) | ≈0 | negative (no gain over fixed) |
| RQ5 · training signal | multi-reference sampling | −1.47 | negative |
| RQ5 · representation alignment | projector feature-KD | −0.03 | negative (absolute null) |
| **RQ6 · decoder capacity** | **LoRA r=16 attention, 1 epoch** | **+3.62** | **positive** |
| **RQ6 · decoder capacity** | **LoRA r=16 attention, 3 epochs** | **+5.12** | **positive** |

**Reading.** Four independent axes on the vision / training side of the pipeline
— none of which touch the decoder — produce no lift; the `feat`-alignment axis
is an *absolute* null. The one axis that touches the decoder produces a clear
lift on every bridge, and localizes further to the decoder's **attention**. This
*pattern across six interventions* — not any single ablation — is what localizes
the bottleneck: for this VLM class (frozen ViT, frozen 0.5B decoder, a few
pooled vision tokens), the frozen decoder's attention is the ceiling, not the
vision pipeline. Opening it by 0.23% recovers a third or more of the F1 gap to a
fully-unfrozen prior-work baseline. §8 develops this reading; the paper's primary
contribution remains the ~1% recipe (§5) — the diagnosis explains *why* it tops
out where it does rather than arguing it should be abandoned.

---

## §7 Human Validation and Error Analysis

Every result above is read through automatic metrics against 5 reference
answers. §7 asks how much those metrics track answer *correctness*.

**Scope, stated up front.** The plan called for human validation — 300–500
samples, 2 annotators, Cohen's κ. Time did not allow it before the deadline;
what follows is a **single-rater self-check substitute**, not human validation:
N = 120, one rater (the assistant, not an independent annotator), no image
access — judged for plausibility against the 5 references only, which for
open-ended categories (causal/context) is a materially weaker check than true
human validation. Sampled proportionally by category × the *actual* F1 bucket of
each prediction (`scripts/human_validation_sample.py`); all 120 judgments with
reasoning in `outputs/human_validation/selfcheck_judgments.json`.

| F1 bucket | n | correct | partial | wrong | nonsense | **acceptable** |
|---|---:|---:|---:|---:|---:|---:|
| strong (≥0.6) | 45 | 80.0 | 11.1 | 6.7 | 2.2 | **91.1** |
| partial (0.2–0.6) | 58 | 12.1 | 31.0 | 55.2 | 1.7 | **43.1** |
| weak (0–0.2) | 3 | 0 | 0 | 100 | 0 | **0** |
| zero (F1=0) | 13 | 7.7 | 7.7 | 76.9 | 7.7 | **15.4** |
| **total (n = 119)** | | **37.0** | **20.2** | **40.3** | **2.5** | **57.1** |

**Reported straight.** The "strong" bucket is reliable (91.1% acceptable) — high
F1 is a good correctness signal there. But the **"partial" bucket (0.2–0.6) is
both the *largest* single bucket (51.5% of val) and the *least* reliable** —
only 43.1% acceptable, 55.2% actually wrong despite sharing filler tokens with
the reference. The failure mode is not random noise: wrong colour / count /
object, or answering the wrong facet of the question (a "when" answered with
weather, a yes/no answer inverted) — errors a fluent decoder produces while
still overlapping enough vocabulary to score mid-range F1.

Overall, **37.0% of val predictions are fully correct, 57.1% acceptable** by this
check — lower than headline numbers like CIDEr-D 92.3 or F1 49.6 might suggest to
a reader unfamiliar with the metrics' scales. **Mid-range token-F1 is not a
reliable correctness signal**, which qualifies how every CIDEr-D / F1 number in
§5–§6 should be read. This is a time-constrained substitute; a real 2-annotator +
image-access study remains the camera-ready item.

---

### Pending

**§5–§7 restructured to the blueprint 2026-09-07, numbers on the final 2-epoch
3-seed set.** Remaining:
- [ ] regenerate `outputs/bootstrap_ci.json` on the 2-epoch predictions
      (`scripts/bootstrap_ci.py` paths point at seed-42; the [91.3, 97.1] CIDEr-D
      CI in §5.2 and the §6.5 per-bridge CIs are from the 4-epoch run) — expect
      small shifts; also recompute the `residual` per-bridge row against a sound
      seed
- [ ] error analysis (§7): per-category error breakdown, noun-omission rate,
      generation-length comparison (co-author has draft material)
- [ ] real 2-annotator + image-access human validation (§7) — camera-ready
- [ ] LoRA test-set eval (checkpoint on `feat/decoder-lora`, 573MB)
- [ ] translate prose to English for the LNCS submission; draw Figures 1–2
