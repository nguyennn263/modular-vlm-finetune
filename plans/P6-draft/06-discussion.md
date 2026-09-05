# 6. Discussion & Limitations

## 6.1 Four axes, one positive: localizing the F1 ceiling to the decoder

§5.3–§5.6 push on four different levers that could, in principle, close the
remaining F1 gap to ViMoE-VQA on top of the frozen-backbone, 1-tile bridge —
visual-compute allocation, training target, representation alignment, and
decoder capacity. §5.7 lays the result out as a single table; three axes are
negative and one is clearly positive, and the shape of that split is itself
the finding:

1. **Reasoning type does not predict visual-compute demand (§5.3–5.4).** Per
   category, the effect of `n_tiles` on answer quality is not significant in
   any of the eight categories (paired bootstrap CIs all include zero); no
   learned policy — reasoning-type-informed or not — beats a fixed `multi_token
   |t1` on held-out test. This holds on both the original and the
   tile-count-augmented retrained checkpoints (§5.3's C3 re-sweep), so it is not
   an artifact of the bridge never having seen >1 tile during training.

2. **Multi-reference training and projector-alignment KD do not lift F1
   (§5.5).** Neither training on a resampled reference each epoch nor
   distilling the bridge toward Vintern's own pre-aligned `mlp1` projector
   moves F1 upward; both leave it flat or slightly negative. The bridge is
   already close to CE-optimal (lowest val CE of the five architectures,
   §5.1) — there is little room for a training-signal or alignment tweak to
   improve on.

3. **Decoder-LoRA is the one intervention that moves F1, and it is
   bridge-agnostic (§5.6).** Opening ~2% of Qwen2-0.5B's parameters (LoRA
   r=16 on `q/k/v/o`) lifts F1 +3.4 on `multi_token` (3/3 seeds locked) and
   +5.4 on `qformer` — a *larger* effect on a different bridge, which rules out
   "one bridge's LoRA got lucky" as an explanation. Both lifts come with lower
   validation CE, and the corpus-level (pycocoevalcap) rescore confirms the
   gain survives the metric-implementation change (§5.6): LoRA'd `multi_token`
   now beats ViMoE-VQA on CIDEr-D/BLEU-4/ROUGE-L by wider margins than the
   plain bridge already did.

**Reading.** Three independent axes on the vision/training side of the
pipeline — none of which touch the decoder — produce no lift; the one axis
that does touch the decoder produces a lift on every bridge it was tried on.
That pattern is more informative than any single ablation: it is not that we
tried one clever trick and it happened to work, it is that *only* the trick
which adds decoder capacity worked, regardless of which trick or which bridge.
For this VLM class — frozen ViT, frozen small (0.5B) decoder, a few pooled
vision tokens — the frozen decoder is the ceiling on token-level phrasing
match, not the vision pipeline. The pooled bridge already discards per-tile
detail the decoder has no way to exploit even when present (§5.3); more
training signal or a better-aligned representation has nothing further to
attach to once the bridge is already CE-optimal (§5.5); and the one lever that
is *not* about what the decoder is given, but about what the decoder itself
can do with it, is the one that moves the needle (§5.6). We report LoRA as a
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
  are essentially unchanged from the random-split table, so prior AutoViVQA
  bridge results were not inflated by leakage.
- The **multi-token bridge** result — corpus CIDEr-D 94.4 / BLEU-4 19.6 /
  ROUGE-L 50.0 on val, above ViMoE-VQA on all three, at 0.78 % trainable
  parameters — holds across 4 seeds (mean±std 92.8±1.1 / 19.2±0.3 / 49.2±0.5),
  a clean, reproducible positive (§5.1).
- The **compute-efficiency characterisation** (§5.2) is the FLOPs/latency
  analysis ViMoE-VQA explicitly deferred: the 1→6-tile lever is real (FLOPs
  ×6.0, latency ×4.0) and the headline bridge spends none of it.
- The **null result on adaptive visual-compute routing** (§5.3–5.4) is
  cross-validated on two independently-trained checkpoint generations
  (n_tiles=1-only and tile-count-augmented) — the same conclusion both times
  rules out "the bridge just never learned to use tiles" as a confound.
- The **decoder-LoRA reference point** (§5.6) is now the most-replicated single
  number in the paper: 3/3 seeds on `multi_token` (std ≈0.03 F1) plus an
  independent second bridge (`qformer`, a larger effect), both confirmed at
  corpus level with `scripts/rescore_corpus.py` verified against the already-
  locked §5.1 row before being trusted for a new one. Three different checks
  (seed, bridge, metric implementation) all agree the effect is real.
- Taken together, §5.7's four-axis table is the load-bearing summary for §6.1:
  it is not one ablation but a *pattern across four independent interventions*
  that localizes the bottleneck, which is more robust to any single
  intervention's idiosyncrasies than any one row would be alone.

## 6.4 Limitations

1. **Single seed** for bridge and policy runs. A 5-seed protocol is needed for
   the camera-ready; current claims rest on paired bootstrap over large
   evaluation splits.
2. **Frozen 0.5 B decoder.** The negative routing/training/alignment results
   (§5.3–5.5) may not generalise to VLMs with a trainable or larger decoder
   that *can* use extra visual tokens — and §5.6's LoRA result is direct
   evidence for exactly that: once the decoder is even slightly unfrozen, part
   of the gap closes. We claim the §5.3–5.5 negatives for the frozen-backbone
   / small-bridge regime only; §5.6 is reported as a reference point, not
   pursued as a multi-seed, multi-config main result.
3. **Decoder-LoRA is undersampled relative to the main spine.** Rank 16 is
   locked at 3 seeds (`multi_token`) and 1 seed on a second bridge
   (`qformer`); a rank sweep (r=8/32) was launched to turn the single r=16
   point into an ablation but had not landed as of this draft. 1 epoch only.
   No corpus-rescore yet on the r=8/32 configs. Treat the *direction and
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
