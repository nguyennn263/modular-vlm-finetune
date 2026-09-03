# 2. Related Work

## 2.1 Adaptive visual computation in VLMs

Reducing the vision-side cost of VLMs has been approached from several angles.
**Visual token pruning / merging** (e.g. token pooling, ToMe-style merging,
FastV, PruMerge) drops or fuses patch tokens after encoding, cutting decoder
cost but not encoder cost. **Early-exit and layer-skipping** in the vision
encoder trade depth for latency. **Mixture-of-resolution / dynamic resolution**
methods (e.g. adaptive tiling in InternVL-style pipelines, LLaVA-NeXT dynamic
resolution) vary how many high-resolution tiles an image is split into. Our
`n_tiles` axis is exactly this last lever, applied to a frozen InternViT-300M
encoder.

What these methods share is that the routing signal is *model-internal*: a
learned gate reads intermediate activations, attention scores, or a small
auxiliary head on the visual features. None of them uses an *external, explicit
label for the type of reasoning the question requires*. Our study isolates
precisely that comparison: reasoning-type supervision versus cheap
model-internal visual features, against a common oracle.

## 2.2 Reasoning-aware routing and MoE for VQA

Sparse mixture-of-experts (MoE) has been proposed for VQA as a way to let
different questions use different sub-networks. **ViMoE-VQA** [cite — KES 2026]
is the most directly relevant: a generative Vietnamese VQA model with a frozen
CLIP ViT-B/32 encoder and frozen PhoBERT text encoder, two transformer fusion
layers, a four-expert Top-2 noisy-gated MoE (Vision / Text / Multimodal /
Specialized experts), and a six-layer autoregressive decoder, trained with a
cross-entropy plus load-balancing objective. ViMoE-VQA reports that its router
"implicitly captures both visual and linguistic cues, enabling approximate
reasoning-aware expert selection."

We take this claim as our starting hypothesis and test it explicitly. Two
observations motivate scrutiny. First, ViMoE-VQA's own leave-one-out ablation
shows that removing *any* single expert degrades BLEU by only 0.11–0.44 points,
and all configurations activate all experts — i.e. the experts are not strongly
specialised. Second, the paper reports no analysis linking expert routing to
question type or reasoning type; the "reasoning-aware" property is asserted, not
measured. Our oracle analysis (§5) supplies the missing measurement, on the same
benchmark, and finds that reasoning type does not predict which visual-compute
action is optimal.

ViMoE-VQA also explicitly defers "a detailed system-level characterization
(FLOPs, latency, memory consumption)" as future work. Our §5.5 compute-efficiency
table fills that gap for the bridge-based frozen-backbone setting.

## 2.3 Offline / oracle-guided policy learning

Learning a routing policy from a pre-computed oracle over a discrete action set
is a form of offline policy learning / imitation of an oracle. We do not use
reinforcement learning or bandit exploration: for every training question we
exhaustively evaluate all actions once, then train the policy by supervised
classification against the utility-maximising action a\*(x, λ) at each cost
trade-off λ. This makes the study fully reproducible and removes exploration
variance as a confound, at the cost of an expensive one-time sweep (§4.4).

## 2.4 Vietnamese VQA and the AutoViVQA benchmark

AutoViVQA [cite — arXiv 2603.09689] provides 19,411 images and 37,077 questions,
each with five diverse free-form Vietnamese answers, annotated with a
reasoning-type label. Prior results on the benchmark include ViT5+ViT,
BARTPhoBEiT, a fine-tuned Vintern-1B, several proprietary LLMs (GPT-5,
Gemini 2.0/2.5 Flash, Llama 3.2), and ViMoE-VQA. AutoViVQA ships only an
80/20 train/val division with no public test split; we construct a grouped
70/15/15 split (§4.1) so that no image appears in more than one split, closing
a caption/context leakage path that a random question-level split leaves open.
