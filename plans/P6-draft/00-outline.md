# Paper 3 — P6 draft outline (LNCS, ~12–15 pp)

**Working title:** *Does Explicit Reasoning-Type Supervision Improve Visual-Computation Allocation? A Controlled Study on Vietnamese VQA*

**Venue:** Trust4NLP special session @ ACIIDS 2027. Deadline 2026-09-27. Springer LNCS/LNAI.

**Thesis (research question, not a claim of invention):**
> We investigate whether explicit reasoning-type supervision can improve the allocation of visual computation in a frozen-backbone VLM beyond model-internal signals alone.

**Answer (headline):** No. On AutoViVQA, an oracle analysis shows reasoning type does not predict per-category visual-compute demand; per-sample routing headroom is dominated by CIDEr measurement noise over near-tied actions; and no learned policy — with or without reasoning-type features — beats the trivial fixed policy "best bridge, minimum tiles". A secondary, positive finding: the multi-token bridge on a leak-free grouped split reaches corpus CIDEr-D 0.94, above the ViMoE-VQA MoE baseline (0.887).

## Section ownership

| § | Title | Owner |
|---|---|---|
| 1 | Introduction + numbered contributions | **me** |
| 2 | Related Work (adaptive visual computation; ViMoE positioning) | **me** |
| 3 | Method (pipeline, bridges, router, policy, action space, oracle objective) | **me** |
| 4 | Experimental Setup (dataset, split, metrics, hardware, GPU-hours, seeds) | **me** |
| 5.1 | Bridge baseline comparison (leak-free split vs prior models) | **me** |
| 5.2–5.8 | Oracle analysis, policy ablation ladder (\|A\|=6, \|A\|=9 held-out), efficiency, stats | **peer** (`05-results.md`) |
| 6 | Discussion & Limitations | **me** |
| 7 | Conclusion | **me** |

## Rigor checklist (Paper 1&2 "8 signatures")

1. ✅ numbered contribution list — §1
2. ✅ formal statement of core mechanism — §3.4 (oracle objective, policy)
3. ✅ dataset statistics table — §4.1 (`outputs/dataset_stats.json`)
4. ⏳ detailed hyperparameter + hardware setup — §4.4 (GPU-hours tally pending)
5. ✅ multi-baseline result table — §5.1
6. ✅ multi-level ablation — §5 (fixed sweep / random / visual-only / rt-only / ours / oracle-cognitive-prior / oracle)
7. ⏳ multi-seed stats (mean±std, CI) — bridge table currently seed 42 only; **need 5 seeds (42,123,3407,2026,8668) to match ViMoE** OR justify single-seed + paired bootstrap
8. ⏳ human validation + quantitative error analysis — NOT DONE (needs user: 300–500 samples, 2 annotators, Cohen's κ)

## Open items blocking a final draft

- [ ] |A|=9 TRAIN split → peer re-runs policy ×3 (more data, less overfit)
- [ ] mt-retrain (tiled multi_token) sweep → confirms/refutes "headroom is noise"
- [ ] 5-seed bridge runs OR a defensible single-seed argument
- [ ] human validation + error analysis (user)
- [ ] final GPU-hours tally
- [ ] P1 FLOPs/latency numbers (compute-efficiency table — fills the gap ViMoE explicitly deferred)
