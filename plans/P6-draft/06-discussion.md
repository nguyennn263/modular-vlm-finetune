# 6. Discussion & Limitations

## 6.1 Why reasoning-type supervision does not help here

Three findings compound (details in §5):

1. **Reasoning type does not predict visual-compute demand.** Per category, the
   effect of `n_tiles` on answer quality is not significant in any of the eight
   categories (paired bootstrap CIs all include zero). A counting question does
   not reliably benefit from more tiles more than a yes/no question does.

2. **Per-sample oracle headroom is measurement noise.** The oracle's apparent
   per-sample advantage comes from taking an argmax over nine actions whose true
   quality is near-tied; per-sample CIDEr variance then decides the argmax. A
   policy trained and evaluated on the *same* split memorises this noise
   (reaching oracle-level M), but the val and test oracle-action distributions
   disagree (val: mostly qformer|t1; test: mostly multi_token|t1), and no policy
   trained on one transfers to the other.

3. **No learned policy beats a fixed one.** On held-out test, `fixed:
   multi_token|t1` — the best bridge at the cheapest tile count — is not beaten
   by `ours`, `visual_only`, or `rt_only`; all learned arms are significantly
   *worse* (paired bootstrap). Adding reasoning-type features on top of visual
   features changes nothing.

The cleanest interpretation: for this VLM class — frozen encoder, frozen small
decoder, a few pooled vision tokens — the visual-compute lever is genuinely flat.
The pooled bridge discards per-tile detail; the frozen 0.5 B decoder cannot
exploit extra visual tokens even when present. There is little allocation
decision to make, so no router — however informed — can add value over "always
use the best configuration".

## 6.2 Relation to ViMoE-VQA's "reasoning-aware" claim

ViMoE-VQA attributes part of its MoE gain to "approximate reasoning-aware expert
selection". Our result does not contradict ViMoE's accuracy numbers, but it does
suggest the mechanism is mis-attributed: on the same benchmark, reasoning type
carries no signal about the *optimal* visual-compute action, and ViMoE's own
leave-one-out ablation shows its experts are not strongly specialised. A more
parsimonious account of the MoE gain is added capacity / ensembling, not
reasoning-type routing. Testing this on ViMoE directly (does expert activation
correlate with question type?) is the obvious follow-up and requires only its
router logs.

## 6.3 What is robust

- The **grouped split** closes an image-level leakage path; our bridge numbers
  are essentially unchanged from the random-split table, so prior AutoViVQA
  bridge results were not inflated by leakage.
- The **multi-token bridge** result — corpus CIDEr-D 94.4 / BLEU-4 19.6 /
  ROUGE-L 50.0 on val, above ViMoE-VQA on all three, at 0.78 % trainable
  parameters — is a clean, reproducible positive.
- The **compute-efficiency characterisation** (§5.5) is the FLOPs/latency
  analysis ViMoE-VQA explicitly deferred.

## 6.4 Limitations

1. **Single seed** for bridge and policy runs. A 5-seed protocol is needed for
   the camera-ready; current claims rest on paired bootstrap over large
   evaluation splits.
2. **Frozen 0.5 B decoder.** The negative routing result may not generalise to
   VLMs with a trainable or larger decoder that *can* use extra visual tokens.
   We claim the result for the frozen-backbone / small-bridge regime only.
3. **`n_tiles ∈ {1,3,6}` and three bridges.** A finer action grid, or bridges
   that consume per-tile tokens without pooling, might expose structure this
   study cannot see. The oracle sweep cost bounds how fine the grid can be.
4. **`M(a;x)` = per-sample CIDEr.** Per-sample CIDEr is noisy for 4-word answers;
   this noise is itself part of finding (2), but a less noisy per-sample quality
   signal (e.g. an LLM judge) could in principle recover some headroom.
5. **CIDEr scale / cross-paper metric implementations.** We standardise on
   pycocoevalcap corpus metrics; baseline rows are as-reported and may use
   different implementations, especially for METEOR and BLEU.
6. **Oracle-sweep subset** is equal-per-category, not proportional; overall
   numbers are re-weighted but tail categories are over-represented in the raw
   sweep.
7. **Human validation and quantitative error analysis** are not yet included
   (planned: 300–500 samples, two annotators, Cohen's κ).

## 6.5 Ethical / reproducibility notes

All models are frozen public checkpoints; only small bridges/heads are trained.
The split script, action space, and oracle tables are released. No human-subjects
data beyond the public AutoViVQA benchmark.
