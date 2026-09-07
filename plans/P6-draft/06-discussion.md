# 6. Discussion & Limitations

## 6.1 Four axes, one positive: localizing the F1 ceiling to the decoder

§6.2–§6.5 push on four different levers that could, in principle, close the
remaining F1 gap to ViMoE-VQA on top of the frozen-backbone, 1-tile bridge —
visual-compute allocation, training target, representation alignment, and
decoder capacity. §6.6 lays the result out as a single table; four axes are
negative and only the decoder axis is positive — and even there, only when
the LoRA is placed on the decoder's *attention* projections. The shape of that
split is itself the finding:

1. **Reasoning type does not predict visual-compute demand (§5.3–5.4).** Per
   category, the effect of `n_tiles` on answer quality is not significant in
   any of the eight categories (paired bootstrap CIs all include zero); no
   learned policy — reasoning-type-informed or not — beats a fixed `multi_token
   |t1` on held-out test. This holds on both the original and the
   tile-count-augmented retrained checkpoints (§5.3's C3 re-sweep), so it is not
   an artifact of the bridge never having seen >1 tile during training.

2. **Multi-reference training and projector-alignment KD do not lift F1
   (§6.4).** Neither training on a resampled reference each epoch (ΔF1 −1.5)
   nor distilling the bridge toward Vintern's own pre-aligned `mlp1` projector
   (ΔF1 −0.03 — an *absolute* null: no measurable effect either way) moves F1
   upward. The bridge is already close to CE-optimal (lowest val CE of the five
   architectures, §6.1) — there is little room for a training-signal or
   alignment tweak to improve on.

3. **Decoder-LoRA is the one intervention that moves F1, it is bridge-agnostic,
   and it works only on the attention projections (§6.5).** Adapting 0.23% of
   Qwen2-0.5B's parameters (LoRA r=16 on `q/k/v/o`, 2.16M) lifts F1 +3.6 on
   `multi_token` (3/3 seeds, std 0.07) and +5.9–7.8 on the four other bridges —
   a *larger* effect on the weaker bridges, which rules out "one bridge's LoRA
   got lucky". After LoRA all five bridges collapse into a 0.6-point F1 band
   (52.6–53.2) and a 2.6-point CIDEr-D band (100.8–103.0), from plain spreads of
   ~4.4 F1 / ~13 CIDEr-D. Crucially, moving the same LoRA budget to the decoder's
   feed-forward layers (`gate/up/down_proj`) *diverges* training (val loss 3–4
   vs 1.37, F1 ~20); the useful headroom is specifically in attention, not the
   decoder broadly. (We flag the MLP result as possibly a hyperparameter
   artifact — α=32 is aggressive for the larger intermediate dimension — so the
   claim is scoped to the recipe's settings.)

**Reading.** Four independent axes on the vision/training side of the
pipeline — none of which touch the decoder — produce no lift (representation
alignment is a flat zero); the one axis that does touch the decoder produces a
lift on every bridge, and only through its attention projections. That pattern
is more informative than any single ablation: it is not that we tried one clever
trick and it happened to work, it is that *only* the trick which adds
attention capacity to the decoder worked, regardless of which bridge.
For this VLM class — frozen ViT, frozen small (0.5B) decoder, a few pooled
vision tokens — the frozen decoder is the ceiling on token-level phrasing
match, not the vision pipeline. The pooled bridge already discards per-tile
detail the decoder has no way to exploit even when present (§5.3); more
training signal or a better-aligned representation has nothing further to
attach to once the bridge is already CE-optimal (§5.5); and the one lever that
is *not* about what the decoder is given, but about what the decoder itself
can do with it, is the one that moves the needle (§6.5). We report LoRA as a
reference point quantifying that ceiling, not as a replacement for the paper's
main spine (§5.1–5.2) — the frozen, 0.78%-param bridge remains the primary
contribution, and this finding explains why that architecture class tops out
where it does rather than arguing it should be abandoned.

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
  are essentially unchanged from the random-split table (≤0.5 F1 / ≤3 CIDEr-D),
  so prior AutoViVQA bridge results were not inflated by leakage. *Caveat:* one
  early Residual-bridge run used in the first comparison was a training-
  instability outlier (val CE 2.35 vs ~1.5–1.7 elsewhere); with a sound 3-seed
  run the Residual numbers rise sharply (F1 37.6→45.6, CIDEr-D 56.3→81.1) — so
  the "barely changed" claim holds for the four stable bridges, not for that
  single broken Residual run.
- The **multi-token bridge** result — corpus CIDEr-D 92.3 / BLEU-4 18.9 /
  ROUGE-L 48.9 on val, above ViMoE-VQA on all three, at 0.78 % trainable
  parameters — holds across 4 seeds (mean±std 92.3±0.6 / 18.9±0.3 / 48.9±0.1)
  **and on held-out test** (F1 49.2 vs val 49.6; CIDEr 93.2 vs 96.5), a clean,
  reproducible positive with no val-set overfitting (§5.1–5.2).
- The **compute-efficiency characterisation** (§5.2) is the FLOPs/latency
  analysis ViMoE-VQA explicitly deferred: the 1→6-tile lever is real (FLOPs
  ×6.0, latency ×4.0) and the headline bridge spends none of it.
- The **null result on adaptive visual-compute routing** (§6.3) is
  cross-validated on two independently-trained checkpoint generations
  (n_tiles=1-only and tile-count-augmented) — the same conclusion both times
  rules out "the bridge just never learned to use tiles" as a confound.
- The **decoder-LoRA result** (§6.5) is the most-replicated finding in the
  paper: +3.6 F1 on `multi_token` (3/3 seeds, std 0.07) and +5.9–7.8 on four
  other bridges (three token-mixing designs), all converging to a 0.6-point F1
  band; confirmed at corpus level and on held-out test. Three checks (seed,
  bridge, metric implementation) plus the attention-vs-MLP contrast all agree.
- Taken together, §6.6's six-axis table is the load-bearing summary for §6.1:
  it is not one ablation but a *pattern across independent interventions* that
  localizes the bottleneck, more robust to any single intervention's
  idiosyncrasies than any one row would be alone.

## 6.4 Limitations

1. **3-seed, not 5.** Bridge and negative-axis runs are 3 seeds (`multi_token`
   4); routing/oracle runs are seed 42. A 5-seed protocol is the camera-ready
   target; current claims rest on 3-seed means (F1 std 0.07–0.94) plus paired
   bootstrap over the full val/test splits.
2. **Frozen 0.5 B decoder.** The negative routing/training/alignment results
   (§6.2–6.4) may not generalise to VLMs with a trainable or larger decoder
   that *can* use extra visual tokens — and §6.5's LoRA result is direct
   evidence for exactly that: once the decoder's attention is even slightly
   unfrozen, part of the gap closes. We claim the §6.2–6.4 negatives for the
   frozen-backbone / small-bridge regime only.
3. **Decoder-LoRA epoch and rank grid is coarse.** r=16 attention is 3 seeds ×
   5 bridges; epoch curve is 1 and 3 (the 5-epoch run was cut at the quota cap
   at ~4 epochs). The MLP / attn+MLP divergence (§6.5) uses the recipe's α=32
   and lr — a lower α or longer schedule might make MLP LoRA viable; we scope
   "attention only" to these settings. Treat the *direction and
   bridge-agnosticism* of the effect as solid, the exact magnitude as
   provisional.
4. **`n_tiles ∈ {1,3,6}` and three bridges.** A finer action grid, or bridges
   that consume per-tile tokens without pooling, might expose structure this
   study cannot see. The oracle sweep cost bounds how fine the grid can be.
5. **`M(a;x)` = per-sample CIDEr.** Per-sample CIDEr is noisy for 4-word answers;
   this noise is itself part of finding (2), but a less noisy per-sample quality
   signal (e.g. an LLM judge) could in principle recover some headroom.
6. **CIDEr scale / cross-paper metric implementations.** We standardise on
   pycocoevalcap corpus metrics; baseline rows are as-reported and may use
   different implementations, especially for METEOR and BLEU.
7. **Oracle-sweep subset** is equal-per-category, not proportional; overall
   numbers are re-weighted but tail categories are over-represented in the raw
   sweep.
8. **Human validation is done only in reduced form.** The plan called for
   300–500 samples, 2 annotators, Cohen's κ; time did not allow it before the
   deadline. §5.8 substitutes a single-rater (assistant) self-check, N=120,
   no image access — a materially weaker check, stated as such where it is
   reported. Its finding — that F1's "partial" bucket (0.2–0.6), the largest
   single bucket, is only 43.1% semantically acceptable despite non-zero
   token overlap — is informative but rests on one rater's judgment without
   an image; a real 2-annotator protocol with image access remains the open
   item for a camera-ready version, and would be needed to make a stronger
   claim than "this is suggestive, not confirmed."

## 6.5 Ethical / reproducibility notes

All models are frozen public checkpoints; only small bridges/heads are trained.
The split script, action space, and oracle tables are released. No human-subjects
data beyond the public AutoViVQA benchmark.
