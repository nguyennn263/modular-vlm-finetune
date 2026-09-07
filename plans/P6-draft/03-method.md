# 3. Method

## 3.1 Frozen-backbone VLM with a trainable bridge

Our base model follows Vintern-1B: a vision encoder **InternViT-300M** and a
language model **Qwen2-0.5B**, connected by a projector. Unlike Vintern-1B's own
downstream-adaptation recipe (full fine-tuning of the encoder and projector, LoRA
on the decoder), **we freeze both backbones** and replace only the projector with
a trainable **bridge** module. Training optimises the bridge alone, by
cross-entropy on a reference answer.

```
Image (1..T tiles, 448×448)
   │   T InternViT forward passes            ← dominant visual FLOPs
   ▼
InternViT-300M  (frozen)  →  T · 256 patch tokens
   │
   ▼
Bridge  (trainable, 0.4–6.9 % of params)  →  k vision tokens
   │
   ▼
Qwen2-0.5B  (frozen, optional rank-16 attention LoRA)  →  Vietnamese answer
```

We study five bridges spanning a capacity ladder from a one-token linear
projector to a 16-query Q-Former (Table 1). The **multi-token bridge** — eight
mean-pooled tokens, one anchor plus seven semantic — is our default: it has the
lowest trainable-parameter count among the multi-token designs (0.78 %) and the
lowest validation cross-entropy of all five.

| Bridge | k (vision tokens) | Trainable params | Mechanism |
|---|---|---|---|
| Residual | 1 | 4.86 M (0.52 %) | linear projector + LayerNorm/GELU residual branch |
| **Multi-Token** | 8 | 7.35 M (0.78 %) | mean-pool patches → 8 tokens (1 anchor + 7 semantic) |
| Tile-Attention | 8 | 4.14 M (0.44 %) | dense patch self-attention, then pool |
| Light Q-Former | 8 | 27.6 M (2.87 %) | 8 learned queries, 2-layer cross-attention |
| Full Q-Former | 16 | 69.4 M (6.91 %) | 16 learned queries, 4 layers, image–text fusion |

## 3.2 Two knobs, one diagnostic ladder

Given a frozen backbone, a small additional parameter or compute budget can be
spent on the **vision side** or the **language side** of the pipeline. We work
through six research questions, one lever at a time, each measured as ΔF1 against
a fixed anchor (the multi-token bridge, one tile, no LoRA):

| | Question | Lever (§) |
|---|---|---|
| **RQ1** | Does a larger / more expressive bridge close the token-F1 gap? | bridge capacity (§3.1, §6.1) |
| **RQ2** | Is the bridge itself the constraint, or is it already near-optimal for the frozen decoder? | validation CE across bridges (§6.1) |
| **RQ3** | Does more visual resolution — more image tiles — help? | `n_tiles` (§3.3, §6.2) |
| **RQ4** | Does *adaptive*, per-question visual-compute allocation help, and can question-type supervision drive it? | router + policy (§3.3, §6.3) |
| **RQ5** | Does a richer training signal or explicit representation alignment to the decoder help? | multi-reference sampling, projector-KD (§3.4, §6.4) |
| **RQ6** | Does adding a small amount of capacity to the *frozen decoder* help, and where in the decoder? | attention vs. feed-forward LoRA (§3.5, §6.5) |

RQ1–RQ4 are vision/training-side; RQ5 straddles; RQ6 is the language side. The
shape of the answers — which levers move F1 and which do not — is the paper's
main finding (§6.6).

## 3.3 Adaptive visual computation (RQ3–RQ4)

**Tile lever.** An image is split into `n_tiles ∈ {1, 3, 6}` tiles of 448×448;
each tile is a separate InternViT forward pass, so `n_tiles` is the primary
visual-compute knob (§5.4 characterises its FLOPs/latency cost).

**Router.** Running in parallel with — and far more cheaply than — the vision
encoder, the router produces two signals. A **cognitive prior** P(r | Q): a
PhoBERT head over AutoViVQA's eight reasoning types, seeing the question text
only (validation macro-F1 0.91; since the reasoning-type label is largely
derivable from question surface form, this is effectively a question-pattern
prior). A **cheap visual state** f(I, Q): the InternViT CLS embedding at one tile
(PCA to 64 dims) plus question length and three image scalars (clarity,
occlusion, object density).

**Offline oracle-guided policy.** For a cost trade-off λ, define the oracle
action over the discrete space `a = (n_tiles, bridge)` as
`a*(x, λ) = argmax_a [ M(a; x) − λ·C(a) ]`, where `M(a; x)` is per-sample CIDEr
against the five references and `C(a) = n_tiles / max_tiles`. We evaluate **all**
actions on **every** train/val/test question once (the *oracle sweep*, §4.4),
then train a policy MLP `π_θ(P(r|Q), f(I,Q), λ)` by supervised classification
against `a*` on the training split. Ablation arms differ only in the policy's
inputs: reasoning-type only, visual-state only, both, and fixed / random
baselines. RQ4 reduces to whether any learned policy beats the fixed policy
"best bridge at one tile" on held-out test.

## 3.4 Training-signal and alignment interventions (RQ5)

Two ways to give the bridge a better target without touching the backbones:
**(a) multi-reference answer sampling** — instead of always training toward the
first of the five references, resample a reference each epoch; **(b)
projector-level knowledge distillation** — add a KD term pulling the bridge
output toward Vintern-1B's own pre-aligned `mlp1` projector, at the feature level
(`align-feat`) or the decoder-logit level (`align-logit`), with weight α on the
KD term.

## 3.5 Decoder LoRA as a targeted intervention (RQ6)

Finally, the smallest departure from "backbone frozen": a rank-16 LoRA
(α = 32, 2.16 M parameters, 0.23 % of the model) on the frozen Qwen2 decoder,
trained on top of a fixed bridge. We vary **which modules** the LoRA touches —
the attention projections `{q,k,v,o}`, the feed-forward projections
`{gate,up,down}`, or all seven — to localise where the decoder's useful headroom
lies, and vary the number of LoRA epochs (1, 3) to bound the effect.
