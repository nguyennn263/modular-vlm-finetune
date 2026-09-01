# modular-vlm-finetune

Train a small **bridge module** that maps frozen InternViT vision embeddings into the
frozen Qwen2-0.5B embedding space (base model: `5CD-AI/Vintern-1B-v3_5`). Both the
vision encoder and the language model stay frozen — only the bridge is trained.

Six bridge architectures: `residual`, `multi_token`, `tile_attention`, `mini_qformer`,
`qformer`, `gated_fusion` (see `src/modeling/bridge_modules.py`).

Dataset: **AutoViVQA** (Vietnamese VQA, 5 reference answers/question). Reasoning-type
labels (`category`, 8 classes) come from `final_vqa_dataset.json`.

---

## Setup

```bash
# local (conda)
bash setup.sh && conda activate vlm-bridge
pip install -e ".[dev,metrics]"

# or plain venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,metrics]"
```

Pinned: `torch==2.2.2`, `transformers==4.38.2` (required for Vintern-1B v3_5).

## Data

```bash
python -m src.cli.download            # ~5 GB into data/raw/  (skip on Kaggle)
python -m src.cli.download --texts-only   # just the CSV + labels
```

On **Kaggle**: add the `nguynrichard/auto-vqabest` dataset to the notebook — it is
mounted at `/kaggle/input/` and `src/data/environment.py` finds it automatically.

## Train

```bash
python -m src.cli.train --bridge residual                     # full dataset
python -m src.cli.train --bridge qformer --limit 2000 --epochs 5
python -m src.cli.train --bridge residual --smoke             # 50 samples, 1 epoch
python -m src.cli.train --bridge residual --resume            # newest checkpoint
python -m src.cli.train --bridge residual --dry-run           # print resolved config
```

There is **no implicit sample cap** on any platform. Config precedence:
`configs/train.yaml` < `configs/bridges/<bridge>.yaml` < `--smoke` < CLI flags.

## Evaluate

```bash
python -m src.cli.evaluate --bridge residual \
    --checkpoint checkpoints/residual/best_model.pt --split val
```

## Kaggle

`notebooks/kaggle_runner.ipynb` is the source of truth for the Kaggle kernel
(`nguyennn263/modular-vlm-finetune`). Regenerate + push:

```bash
python notebooks/build_kaggle_runner.py
kaggle kernels push -p notebooks/
```

## Paper 3 pipeline (`plans/final-plan.md`)

Phase orchestrators print each step and its command; `--dry-run` lists without running.

```bash
python scripts/phase0_build_data.py     # labelled table + 70/15/15 grouped split
python scripts/phase1_profile.py        # multi-tile + FLOPs/latency calibration  (TODO)
python scripts/phase2_expA.py           # 6 bridge baselines, 3 seeds
python scripts/phase3_expB.py           # bridge x category fork                  (TODO)
python scripts/phase4_oracle_policy.py  # oracle sweep + router + policy          (TODO)
python scripts/phase5_eval.py           # ablation ladder + Pareto + human eval   (TODO)
```

## Layout

```
src/
  cli/         train.py  evaluate.py  download.py  profile.py
  config/      loader.py
  data/        environment.py  loader.py  collator.py  data_actions.py
               labeled_table.py  split.py
  modeling/    bridge_modules.py  router.py  policy.py
  training/    setup.py  trainer.py
  utils/       logging.py  device.py  paths.py  data_loader_helper.py
scripts/       phase0..5  _phase.py
configs/       train.yaml  data.yaml  action_space.yaml  bridges/*.yaml
metrics/       BLEU / METEOR / ROUGE / CIDEr / WUPS / accuracy / F1 ...
tests/         test_imports.py  test_data_pipeline.py  checks/
legacy/        pre-restructure code (not imported) — see legacy/README.md
```

## Test

```bash
pytest -q          # import smoke + data-pipeline unit tests
```
