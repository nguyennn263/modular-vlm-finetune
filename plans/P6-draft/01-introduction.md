# 1. Introduction

Vision-language models (VLMs) spend most of their inference budget in the vision
encoder: encoding an image at high resolution, or as many tiles, costs far more
than the language decoder's forward pass for a short answer. A natural idea is to
*adapt* that visual budget per question — spend more vision compute on questions
that need it, less on questions that do not. The open design question is what
signal should drive that decision.

One hypothesis, implicit in recent "reasoning-aware" mixture-of-experts VQA
models such as ViMoE-VQA [cite], is that the *type of reasoning* a question
demands is informative: a counting question and a yes/no question plausibly need
different amounts of visual processing. If true, a cheap question-only classifier
that predicts reasoning type could steer visual computation before the expensive
vision encoder even runs.

This paper tests that hypothesis directly. We build a frozen-backbone Vietnamese
VLM in which the only trainable component is a small *bridge* module between a
frozen InternViT-300M encoder and a frozen Qwen2-0.5B decoder, and we define a
discrete action space over (number of image tiles) × (bridge architecture). For
every validation and test question we run an *oracle sweep*: we evaluate every
action and record its answer quality, giving us, per question, the best possible
action and the quality gap between actions. Against this oracle we ask: does a
router that sees explicit reasoning-type supervision allocate visual computation
better than a router that sees only cheap model-internal visual features, or than
a trivial fixed policy?

**The answer is no, on three levels.** (i) Per reasoning-type category, the
number of tiles has no statistically significant effect on answer quality
(paired bootstrap CIs all include zero). (ii) Per sample, the oracle's apparent
routing headroom is an artifact: the argmax over near-tied actions is dominated
by CIDEr measurement noise and does not transfer from validation to test. (iii)
No learned policy — with reasoning-type features, with visual features, or with
both — beats the trivial fixed policy "use the best bridge at the minimum tile
count" on held-out test data.

We report this as a careful negative result. Along the way we also establish a
positive, reproducible finding: on a leak-free grouped split of AutoViVQA, the
multi-token bridge (0.78 % trainable parameters) reaches corpus CIDEr-D 0.94,
above the ViMoE-VQA MoE baseline (0.887) and every other published model on this
benchmark for generation metrics, while trailing on token-level F1.

## Contributions

1. **A controlled instrumentation pipeline** (§3) for studying adaptive visual
   computation in frozen-backbone VLMs: a bridge-only trainable VLM, a discrete
   (tiles × bridge) action space with a normalised cost term, and an offline
   oracle sweep that yields per-sample best-action and quality-gap labels.
2. **A question-only cognitive-prior router** P(r|Q) (PhoBERT, macro-F1 0.91 over
   8 reasoning types) and a cheap visual-state probe f(I,Q) (InternViT CLS
   embedding at one tile, plus metadata features), combined by a policy MLP
   trained by offline oracle-guided learning.
3. **A negative result, rigorously established**: explicit reasoning-type
   supervision does not improve visual-computation allocation beyond
   model-internal signals — nor does any learned policy beat a fixed one — for
   this VLM class on AutoViVQA. We trace this to (a) reasoning type not
   predicting visual-compute demand and (b) per-sample oracle headroom being
   measurement noise over near-tied actions.
4. **A leak-free bridge-architecture benchmark** on AutoViVQA with a grouped
   70/15/15 split (no image shared across splits), including a compute-efficiency
   characterisation (FLOPs, latency, throughput by tile count) that prior work on
   this benchmark explicitly deferred.
