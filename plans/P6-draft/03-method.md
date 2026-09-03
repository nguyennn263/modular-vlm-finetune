# 3. Method

## 3.1 Frozen-backbone VLM with a trainable bridge

Our base model follows Vintern-1B-v3.5: a vision encoder **InternViT-300M** and a
language model **Qwen2-0.5B**, connected by a projector. We freeze both backbones
and replace only the projector with a trainable **bridge** module

```
Image (1..T tiles, 448×448)
   │   T forward passes            ← dominant visual FLOPs
   ▼
InternViT-300M  (frozen)  →  T · 256 patch tokens
   │
   ▼
Bridge  (trainable)  →  k vision tokens        (k depends on bridge)
   │
   ▼
Qwen2-0.5B  (frozen)  →  free-form Vietnamese answer
```

Training optimises only the bridge, by cross-entropy on the reference answer
(the first of the five references; §4.2 discusses answer selection). We evaluate
five bridges spanning the pooled-vs-attentive and few-vs-many-token design axes:

| Bridge | k (vision tokens) | Mechanism |
|---|---|---|
| Residual | 1 | linear projector + LayerNorm/GELU residual branch |
| Multi-Token | 8 | mean-pool patches → 8 tokens (1 anchor + 7 semantic) |
| Tile-Attention | 8 | patch self-attention then pool |
| Light Q-Former | 8 | 8 learned queries, 2-layer cross-attention |
| Full Q-Former | 16 | 16 learned queries, 4 layers, image–text fusion |

## 3.2 Router: cognitive prior and cheap visual state

The router runs in parallel with — and far more cheaply than — the vision
encoder, and produces two signals.

**Cognitive prior P(r | Q).** A PhoBERT-base-v2 encoder with an 8-way
classification head over AutoViVQA reasoning types
{relational, recognition, spatial, causal, action, counting, context, yes/no}.
It sees the **question text only**. Validation macro-F1 is 0.91. Because the
reasoning-type label in AutoViVQA is derivable from question surface form, this
head is effectively a question-pattern prior — a point we return to in §6.

**Cheap visual state f(I, Q).** A low-cost probe computed without a full
multi-tile encode: the InternViT CLS embedding at `n_tiles = 1` (PCA-reduced to
64 dims), question length, and three image-level scalars (clarity, occlusion,
object density). This is the "model-internal signal" baseline against which
reasoning-type supervision is tested.

## 3.3 Action space

An action `a = (n_tiles, bridge)` with

- `n_tiles ∈ {1, 3, 6}` — the number of InternViT forward passes, i.e. the
  primary visual-compute lever;
- `bridge ∈ {multi_token, qformer, mini_qformer}` — the three strongest bridges
  from §5.1 / the bridge-×-category study (§5), giving `|A| = 9`.

A reduced `|A| = 6` space over {qformer, mini_qformer} × {1, 3, 6} is used for the
initial ablation (§5.3); the full `|A| = 9` space adds the multi-token bridge.

## 3.4 Oracle and offline policy learning

**Cost.** `C(a) = n_tiles / max_tiles ∈ (0, 1]` — the normalised number of vision
encoder invocations. Wall-clock latency is measured separately (§5.5) and is not
part of the oracle objective, so that the λ grid stays on a single scale.

**Quality.** `M(a; x)` is the corpus-consistent answer quality of action `a` on
question `x` — here per-sample CIDEr against the five references.

**Oracle.** For a cost trade-off `λ`, the oracle action is

```
a*(x, λ) = argmax_a [ M(a; x) − λ · C(a) ]
```

We sweep `λ ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0}`. For every question in
train / val / test we evaluate **all** actions once and store `M(a; x)` and
`C(a)` (the *oracle sweep*, §4.4).

**Policy.** `π_θ(P(r|Q), f(I,Q), λ) → a` is a small MLP trained by supervised
classification against `a*(x, λ)` on the training split, sampling λ from the grid.
Ablation arms differ only in which inputs π sees:

| Arm | Inputs |
|---|---|
| `ours` | P(r\|Q), f(I,Q), λ |
| `rt_only` | P(r\|Q), λ |
| `visual_only` | f(I,Q), λ |
| `fixed(a)` | none — always action a |
| `random` | none — uniform over A |
| `oracle_cog_prior` | true reasoning type only |
| `oracle` | full `a*(x, λ)` (upper bound) |

The research question reduces to a comparison among `ours`, `rt_only`,
`visual_only`, and the best `fixed(a)` on held-out test data (§5.3–5.4).
