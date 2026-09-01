# legacy/

Code from before the 2026-09 restructure. **Nothing here is imported by live code.**
Kept for reference only — the working equivalents:

| legacy | replacement |
|---|---|
| `scripts/exp{1..6}_*.py`, `scripts/run_all_experiments.py`, `scripts/training_runner.py` | `python -m src.cli.train --bridge <name>` |
| `scripts/base_experiment.py` | `src/cli/train.py` |
| `scripts/collect_ablation_results.py` | `python -m src.cli.evaluate` + aggregation in phase scripts |
| `data/loaders.py`, `data/lazy_loaders.py`, `data/training_data_provider.py`, `data/unified_loader.py`* | `src/data/loader.py` |
| `data/collator_OLD.py`, `data/collator_examples.py` | `src/data/collator.py` |
| `data/dataset.py`, `data/onesample_dataset.py`, `data/finetune_dataset.py` | (trainer wraps `list[OneSample]` directly) |
| `data/processor.py` | tiling lives in `src/training/trainer.py` (`load_image`) |
| `data/download_data.py`, `download_data.sh` | `python -m src.cli.download` |
| `middleware/config.py`, `middleware/config_loader.py` | `src/config/loader.py` |
| `middleware/data.py`, `middleware/monitor.py` | unused (broken imports) |
| `logging_setup.py` | `src/utils/logging.py` |
| `finetune_trainer.py` | `src/training/trainer.py` |
| `setup-gpu.sh`, `configure-device.sh` | `setup.sh` / `setup_kaggle.sh` |
| `configs/*.yaml` | `configs/{train,data,action_space}.yaml`, `configs/bridges/*.yaml` |
| `tests/test_training_pipeline.py` | `tests/test_imports.py`, `tests/test_data_pipeline.py` |

\* `unified_loader.py` was *renamed* to `src/data/loader.py`, not copied here.
