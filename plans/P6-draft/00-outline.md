# Paper 3 — P6 draft outline (LNCS, ~12–15 pp)

**Working title:** *Improving Vintern-1B for Vietnamese VQA on a 1 % Parameter
Budget: A Frozen-Backbone Recipe and a Bottleneck Diagnosis*

**Venue:** Trust4NLP special session @ ACIIDS 2027. Deadline 2026-09-27.
Springer LNCS/LNAI.

**Research question:**
> Can Vintern-1B be improved on AutoViVQA by training only ~1 % of its parameters,
> with the entire backbone frozen — and if that is not enough, where is the
> bottleneck?

**Answer (headline):**
1. **The cheap recipe is competitive.** Frozen InternViT-300M + frozen Qwen2-0.5B
   + a 0.78 % mean-pooled bridge (1 tile) + a 0.23 % rank-16 attention LoRA on the
   decoder matches/beats the fully fine-tuned Vintern-1B on all generation
   metrics and beats ViMoE-VQA on corpus CIDEr-D / BLEU-4 / ROUGE-L, at ~1 %
   trainable params and no backbone fine-tuning. Held-out test matches val.
2. **The bottleneck is the frozen decoder's attention.** A 6-question ablation
   ladder: four vision/training-side levers (bigger bridge, more tiles, adaptive
   routing, multi-ref training, representation alignment) produce no token-F1
   lift; only a decoder LoRA does, on every bridge, and only through the
   attention projections (feed-forward LoRA at the same budget diverges).
3. **Counterpoint to "reasoning-aware" routing:** on the same benchmark,
   reasoning type carries no signal about the optimal visual-compute action, and
   no learned policy beats a fixed one.

## Section map / ownership

| § | Title | File | Owner |
|---|---|---|---|
| 1 | Introduction + contributions | `01-introduction.md` | me ✅ |
| 2 | Related Work | `02-related-work.md` | me ✅ |
| 3 | Method (frozen arch, 5 bridges, 2-knobs × 6-RQ, router/oracle, decoder-LoRA) | `03-method.md` | me ✅ |
| 4 | Experimental Setup | `04-setup.md` | me ✅ |
| 5 | Main Results (recipe vs baselines, corpus + CI, held-out test, compute-efficiency) | `05-results.md` | peer ✅ |
| 6 | Ablation: Hunting the Bottleneck (6.1 bridge → 6.2 tiles → 6.3 routing → 6.4 training/align → 6.5 decoder-LoRA + attention-localization → 6.6 six-axis summary) | `05-results.md` | peer ✅ |
| 7 | Human Validation and Error Analysis | `05-results.md` | peer ✅ |
| 8 | Discussion & Limitations | `06-discussion.md` | me ✅ (consistency pass: peer) |
| 9 | Conclusion | `07-conclusion.md` | me ✅ |
| — | Bridge-comparison detail (SUPERSEDED backup) | `05.1-bridge-baseline.md` | me |

## Rigor checklist

1. ✅ numbered contribution list — §1
2. ✅ formal statement of the mechanism — §3.2 (2-knobs × 6-RQ), §3.5 (LoRA localization)
3. ✅ dataset statistics table — §4.1
4. ✅ hyperparameter + hardware setup — §4.3–4.4
5. ✅ multi-baseline result table — §5.1 (9 baselines)
6. ✅ multi-level ablation — §6 (6-RQ ladder + attention/MLP/attn+MLP LoRA + epoch/rank)
7. ✅ multi-seed stats — 3 seeds (4 for headline), bootstrap CIs (`outputs/bootstrap_ci.json`)
8. ⏳ **human validation + error analysis** — self-check N=120/1-rater done (§7);
   real 2-annotator + Cohen's κ study PENDING (needs user + 1 person)

## Open items before submission

- [ ] `[cite]` → real bibliography (Vintern-1B, ViMoE-VQA KES 2026, AutoViVQA
      arXiv 2603.09689, BLIP-2, LLaVA, Frozen, Inference-Optimal VLMs 2411.03312,
      LoRA, adapters, prefix-tuning, FFN-as-KV-memory)
- [ ] Verify the "Vintern-1B fine-tuned" baseline recipe against AutoViVQA §4
      (currently written as "ViT + projector full fine-tune, LLM LoRA")
- [ ] 3 figures: (a) bridge-equalizing bar (b) tile-collapse line (c) method diagram
- [ ] English consistency pass across §1–9 (peer — in progress)
- [ ] Human validation study (user)
- [ ] Optional after Fri quota reset: clean LoRA 5-epoch, TIER-2 MLP HP retune
