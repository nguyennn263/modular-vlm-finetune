# 7. Conclusion

We asked whether explicit reasoning-type supervision improves the allocation of
visual computation in a frozen-backbone Vietnamese VLM beyond cheap
model-internal signals. Using an oracle sweep over a discrete (tiles × bridge)
action space on AutoViVQA, we find it does not: reasoning type does not predict
per-category visual-compute demand; the per-sample oracle headroom is CIDEr
measurement noise over near-tied actions and does not transfer across splits; and
no learned policy — reasoning-type-aware, visual-feature-aware, or both — beats
the trivial fixed policy "best bridge at minimum tile count" on held-out test.

This is a bounded negative result: it holds for the frozen-encoder /
frozen-small-decoder / pooled-bridge regime, where the visual-compute lever is
empirically flat. It also suggests that "reasoning-aware" routing claims for MoE
VQA models on this benchmark deserve direct measurement rather than assumption.

Two positive contributions stand independently: a leak-free grouped-split
benchmark on which a 7 M-parameter multi-token bridge exceeds the ViMoE-VQA MoE
on corpus CIDEr-D, BLEU-4, and ROUGE-L; and a compute-efficiency characterisation
of the pipeline that prior work left open. The instrumentation — bridge-only VLM,
normalised-cost action space, offline oracle — is released for studying adaptive
visual computation in settings where the lever is less flat.
