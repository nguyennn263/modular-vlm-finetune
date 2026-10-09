# 9. Conclusion

We asked whether Vintern-1B can be improved on AutoViVQA by training only a small
fraction of its parameters — with the entire backbone frozen — and, where that is
not enough, where the bottleneck lies.

**The cheap recipe is competitive.** A frozen InternViT-300M and frozen
Qwen2-0.5B, connected by a 7.35 M-parameter (0.78 %) mean-pooled bridge at a
single image tile, and adapted with a 2.16 M-parameter (0.23 %) rank-16 LoRA on
the decoder's attention projections, matches or beats the fully fine-tuned
Vintern-1B on every generation metric (ROUGE-L +1.0, BLEU +14.9, METEOR +10.0,
CIDEr-D +18.1) and beats the ViMoE-VQA mixture-of-experts on corpus CIDEr-D,
BLEU-4, and ROUGE-L — at roughly 1 % of the trainable parameters, on one 16 GB
GPU, with no backbone fine-tuning. Held-out test numbers match validation within
0.5 F1.

**The bottleneck is the frozen decoder's attention.** Across a six-question
ablation ladder, four independent vision- and training-side levers — a 10×-larger
bridge, more image tiles, a learned per-question routing policy, multi-reference
training, and projector-level representation alignment — produce no improvement
in token-level F1 (alignment is an exact null, ΔF1 −0.03). The only lever that
helps is a small LoRA on the frozen decoder, it helps on every one of the five
bridge architectures, and it works only through the attention projections:
placing the same LoRA budget on the feed-forward layers destabilises training.
Once the attention LoRA is present, the five bridges — spanning 4.1 M to 69.4 M
parameters and three token-mixing designs — collapse into a 0.6-point F1 band.
For this VLM class, the ceiling on token-level phrasing is the frozen decoder's
attention capacity, not the visual pipeline.

**This is also a direct counterpoint to "reasoning-aware" routing.** On the same
benchmark, using AutoViVQA's own reasoning-type labels and an exhaustive oracle
over visual-compute actions, reasoning type carries no signal about which action
is optimal, and no learned policy beats a fixed one. Mixture-of-experts gains on
this benchmark are more parsimoniously explained by added capacity than by
reasoning-type specialisation, and such claims warrant direct measurement.

**Scope and limitations.** The negative vision/training results hold for the
frozen-encoder / frozen-small-decoder / pooled-bridge regime; a trainable or
larger decoder may use extra visual tokens the frozen 0.5 B decoder cannot. The
feed-forward-LoRA divergence is at the recipe's rank and learning rate — a
gentler schedule might make it viable. The human-judgment check on token-F1 is a
single-rater substitute for a full study.

We release the code, the leak-free grouped 70/15/15 split, the per-question
oracle tables over the (tiles × bridge) action space, and all trained bridge and
LoRA checkpoints. The natural next step is to vary the one component we kept
frozen and small — the decoder — and test whether a larger frozen Vietnamese
decoder closes the remaining token-F1 gap without any of the levers that failed
here.
