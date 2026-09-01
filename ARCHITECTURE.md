# Architecture

Cognitive-supervised adaptive visual computation for Vietnamese VQA.
Research question: *does explicit reasoning-type supervision improve the allocation
of visual computation beyond model-internal signals alone?*

---

## 1. Inference-time system

```
                          ┌──────────────────────────────────────────┐
   Image ─────────┬──────▶│  ROUTER  (cheap, runs in parallel)        │
                  │       │                                          │
   Question ──────┼──────▶│  P(r|Q) = PhoBERT-base-v2 + linear head   │──┐ Δ⁸
                  │       │           (question only, 8 categories)   │  │
                  │       │  f(I,Q) = InternViT CLS @ 1 tile (PCA-64) │──┤ ℝ⁶⁴
                  │       │           + q_len + image clarity/occl/   │  │
                  │       │           object-density (no `category`)  │  │
                  │       └──────────────────────────────────────────┘  │
                  │                                                     ▼
   λ  (cost ──────┼───────────────────────────▶ ┌──────────────────────────────────┐
   weight in                                    │  POLICY MLP                       │
   U = M − λ·C)   │                              │  (P(r|Q), f(I,Q), λ) → action     │
                  │                              │  argmax over |A| = 9             │
                  │                              └──────────────────────────────────┘
                  │                                              │
                  │              action a = (n_tiles ∈ {1,3,6},  bridge ∈ top-3)
                  │                                              │
                  ▼                                              ▼
        ┌────────────────────┐                        n_tiles InternViT passes
        │  dynamic tiling    │───────────────┐
        │  (B, T, 3,448,448) │               ▼
        └────────────────────┘      ┌────────────────────────┐
                                    │  InternViT-300M        │  ◀─ FROZEN
                                    │  (frozen)              │     dominant visual cost:
                                    └────────────────────────┘     FLOPs ×6, latency ×4
                                              │  T·256 patch tokens  between 1 and 6 tiles
                                              ▼
                                    ┌────────────────────────┐
                                    │  BRIDGE  (top-3)       │  ◀─ TRAINED (0.4–2.9% params)
                                    │  replaces MLP projector │     the only trained component
                                    └────────────────────────┘
                                              │  k vision tokens
                                              ▼
                                    ┌────────────────────────┐
                                    │  Qwen2-0.5B  (frozen)  │  ◀─ FROZEN
                                    └────────────────────────┘
                                              │
                                              ▼
                                           Answer
```

Base model: `5CD-AI/Vintern-1B-v3_5` (InternViT-300M + Qwen2-0.5B). Only the
**bridge** is trained; the vision encoder and the LLM stay frozen.

### Components

| Component | What it is | Trained? | Code |
|---|---|---|---|
| InternViT-300M | vision encoder, 256 patch tokens/tile at 448px | frozen | (Vintern) |
| Bridge | vision→LLM projector; 5 architectures (see §2) | **yes** | `src/modeling/bridge_modules.py` |
| Qwen2-0.5B | language model | frozen | (Vintern) |
| `P(r|Q)` | reasoning-type prior from the question | **yes** (separate) | `src/modeling/router.PrQHead` |
| `f(I,Q)` | cheap visual-state features, no label | derived | `src/cli/build_fiq.py` |
| Policy | maps `(P(r|Q), f(I,Q), λ)` → action | **yes** | `src/modeling/policy.PolicyMLP` |

### Action space `A = (n_tiles, bridge)` — calibrated in P1

- `n_tiles ∈ {1, 3, 6}` — number of InternViT forward passes. **Primary compute
  lever.** Measured (P100): FLOPs ×6.0, latency ×4.0, throughput ×5.2 between 1
  and 6 tiles.
- `bridge ∈` top-3 from the Exp B fork.
- Cost `C(a) = n_tiles / 6` (normalized). Quality `M(a; x) = CIDEr` vs 5 refs.
- Oracle utility `U(a; x, λ) = M(a; x) − λ·C(a)`,  `a*(x, λ) = argmax_a U`.

---

## 2. The 5 bridges (capacity ladder)

| # | Bridge | Mechanism | vision tokens `k` | trainable params |
|---|---|---|---|---|
| 1 | `residual` | baseline Linear + LayerNorm→FC→GELU→FC residual on the pooled vector | 1 | 4.9 M (0.52%) |
| 2 | `multi_token` | 1 anchor token + k−1 learned tokens from the pooled vector | 8 | 7.3 M (0.78%) |
| 3 | `tile_attention` | self-attention across patch tokens → learned query pool | 8 | 4.1 M (0.44%) |
| 4 | `mini_qformer` | 2 transformer layers, learned queries cross-attend to patches | 8 | 27.6 M (2.87%) |
| 5 | `qformer` | 4 transformer layers + vision/question gated fusion | 16 | 69.4 M (6.91%) |

`gated_fusion` exists in the code but is not in the study (near-duplicate of
`residual`). Prior numbers on the *old* split: `plans/results-5bridge.md`
(multi_token best — CIDEr 99.9, BLEU 16.5, METEOR 41.6).

Only `tile_attention`, `mini_qformer`, `qformer` consume patch tokens, so only
they can exploit `n_tiles > 1`; the pooled bridges get the mean over `T·256`
tokens.

---

## 3. Build pipeline (P0 → P5)

```
P0  DATA
    data/raw/  ──[labeled_table]──▶  data/labeled.parquet          (join final_vqa_dataset.json
               8 category, 36,980 rows, ~200 noise rows dropped     + quality CSV on (img_id,question))
        │
        └──[split]──▶  data/splits/{train,val,test}.jsonl          (70/15/15, grouped by image,
                       25,933 / 5,544 / 5,503 QA                     stratified by category, seed 42, 0 leak)

P1  CALIBRATION                                          ─── verified (kernel v8/v9)
    src.cli.profile  ──▶  outputs/profile/pipeline_cost.json
    => action = (n_tiles, bridge),  grid {1,3,6},  C(a) = n_tiles/6

P2  EXP A  (bridge baselines)                            ─── ⚠ NOT RUN (compute marathon)
    train --bridge <b> --split-dir data/splits --seed <s>   ×5 bridges ×N seeds
    ──▶  checkpoints/expA/seed<s>/<b>/best_model.pt

P3  EXP B  (bridge × reasoning-type fork)                ─── code + tests, runs after P2
    evaluate --split-dir  ×5 bridges  ──▶  eval_val_samples.jsonl   (per-sample CIDEr × category)
    src.analysis.expB  ──▶  heatmap + paired bootstrap(best vs 2nd) + Holm
    ──▶  top-3 bridges  →  edit configs/action_space.yaml:bridges

P4  ORACLE + ROUTER + POLICY
    ├─ train_router (PhoBERT)        ─── verified (v11/v12): macro-F1 ~0.92
    │  ──▶ checkpoints/router/best.pt,  outputs/router/prq_{train,val}.parquet
    ├─ build_fiq (InternViT CLS)     ─── verified (v12)
    │  ──▶ outputs/fiq/{train,val}.parquet
    ├─ oracle sweep                  ─── code + tests, runs after P2/P3
    │  7,500 samples × 9 actions × greedy generate()  →  outputs/oracle/table.parquet  (M, C)
    │  ──[oracle_labels]──▶  outputs/oracle/labels.parquet   (a*(x, λ), 7 λ points)
    └─ train_policy  ×3 arms:
          ours        = P(r|Q) + f(I,Q)
          rt_only     = P(r|Q)             (Reasoning-type-only ablation)
          visual_only =          f(I,Q)    (Visual-state-only ablation)
       ──▶  checkpoints/policy_<arm>/best.pt   (loss = CE vs a*)

P5  EVAL ON TEST                                         ─── code + tests, runs after P4
    oracle sweep TEST  (5,503 × 9)  ──▶  outputs/oracle_test/table.parquet
    src.cli.eval_ladder:  7 arms × 7 λ  →  (mean M, mean C)
        fixed:<action> ×9  │  random  │  oracle (upper bound)  │  ours / rt_only / visual_only
    ──▶  outputs/eval/{ladder.csv, pareto.csv, behaviour.json}
    + compute-efficiency table, human validation (Cohen's κ), error analysis   ─── manual
```

### Signal separation (thesis integrity)

- `P(r|Q)` uses the **question only** and is supervised by `category`.
- `f(I,Q)` uses **cheap visual/metadata signals** and **never** `category`.
- The ablation compares: `rt_only` vs `visual_only` vs `ours` vs `oracle-cognitive-prior`
  (true category) vs `oracle` (knows M).

Caveat to state in the paper: `category` was LLM-generated **from the question**,
so `P(r|Q)` reaches macro-F1 ≈ 0.92 — the "cognitive prior" is a strong but
shallow signal, effectively a *question-pattern* prior.

---

## 4. Repo map

```
src/
  cli/         train · evaluate · train_router · train_policy · oracle · build_fiq · profile · download
  analysis/    stats (bootstrap/permutation/Holm) · expB · oracle (a*/frontier) · ablation (ladder/pareto)
  modeling/    bridge_modules · router (PrQHead, VisualStateProbe) · policy (PolicyMLP)
  training/    setup (VisionLanguageBridge) · trainer (BridgeTrainer, ~1.5k lines)
  data/        environment · loader · collator · tiling · labeled_table · split
  reasoning_types.py     the 8 categories (zero-dep single source of truth)
scripts/       phase0_build_data … phase5_eval   (thin orchestrators, `--dry-run` lists steps)
configs/       train.yaml · data.yaml · action_space.yaml · bridges/*.yaml
assets/        final_vqa_dataset.json.gz          (reasoning-type labels, bundled)
metrics/       BLEU · METEOR · ROUGE · CIDEr · WUPS · accuracy · F1 · precision · recall
tests/         48 unit tests (import smoke · data pipeline · stats · expB · oracle · policy · ablation)
notebooks/     kaggle_runner.ipynb  (+ build_kaggle_runner.py)
plans/         final-plan.md (execution) · next-steps.md · results-5bridge.md · ...
legacy/        pre-restructure code (not imported)
```

### Entry points

```
python -m src.cli.train    --bridge residual --split-dir data/splits [--n-tiles N] [--seed S] [--resume]
python -m src.cli.evaluate  --bridge residual --checkpoint <pt> --split-dir data/splits --split val
python -m src.cli.train_router  --split-dir data/splits --predict-splits train,val
python -m src.cli.build_fiq   --split-dir data/splits --splits train,val
python -m src.cli.oracle    --bridges <top3> --n-tiles 1,3,6 --subset 7500 --ckpt-dir checkpoints/expA/seed42
python -m src.cli.train_policy  --prq ... --labels ... [--features ...] [--no-prq]
python -m src.cli.eval_ladder   --policies ours=...,rt_only=...,visual_only=...
python -m src.cli.profile   --n-tiles 1 2 4 6

python scripts/phase{0..5}_*.py [--dry-run]
```

---

## 5. Status

| Phase | Code | Unit-test | Kaggle-verified | Full run |
|---|:-:|:-:|:-:|---|
| P0  data | ✅ | ✅ | ✅ | ✅ |
| P1  profiler + multi-tile training | ✅ | ✅ | ✅ (v8, v9) | ✅ |
| P2  bridge training CLI | ✅ | ✅ | ✅ | **Exp A marathon — user launches** |
| P3  Exp B fork | ✅ | ✅ (8) | — | after Exp A |
| P4  router / f(I,Q) | ✅ | ✅ | ✅ (v10–v12) | ✅ |
| P4  oracle / policy | ✅ | ✅ (14) | — | after Exp A |
| P5  ablation ladder / Pareto | ✅ | ✅ (7) | — | after Exp A |
| P6  paper | — | — | — | deadline 2026-09-27 |

**Critical path:** Exp A (5 bridges × N seeds on the grouped split) unlocks P3 → P4
oracle → P5. Oracle sweep (~40 GPU-h of `generate()`) is the compute bottleneck;
see `plans/next-steps.md` for the budget and options.
