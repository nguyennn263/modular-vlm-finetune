"""Regenerate notebooks/kaggle_runner.ipynb (source of truth for the Kaggle kernel).

    python notebooks/build_kaggle_runner.py
    kaggle kernels push -p notebooks/          # after editing this file
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines][:-1] + [lines[-1]]}


CELLS = [
    md("# modular-vlm-finetune — Kaggle runner",
       "",
       "Trains one bridge module on top of frozen Vintern-1B on the full AutoViVQA data.",
       "There is **no implicit sample cap** — pass `--limit` / `--smoke` yourself for short runs.",
       "",
       "Sessions cap at ~12h; checkpoints go to `/kaggle/working/checkpoints/` and `--resume` picks up where you left off."),
    code("BRANCH = 'chore/repo-restructure'   # change after merge to main",
         "REPO   = 'https://github.com/nguyennn263/modular-vlm-finetune.git'"),
    code("import os",
         "if not os.path.isdir('/kaggle/working/modular-vlm-finetune'):",
         "    !git clone $REPO /kaggle/working/modular-vlm-finetune",
         "%cd /kaggle/working/modular-vlm-finetune",
         "!git fetch origin && git checkout $BRANCH && git pull"),
    code("!bash setup_kaggle.sh"),
    md("### Data",
       "",
       "The `nguynrichard/auto-vqabest` dataset is attached in the kernel metadata and mounted at",
       "`/kaggle/input/`. `src/data/environment.py` resolves the path automatically — nothing to do."),
    code("# sanity: resolve config + data paths without loading the model",
         "!python -m src.cli.train --bridge residual --dry-run"),
    code("# SMOKE run — proves the pipeline trains + evaluates + checkpoints end to end.",
         "# Swap for the full run below once this is green.",
         "!python -m src.cli.train --bridge residual --smoke --output-dir /kaggle/working/checkpoints"),
    code("!python -m src.cli.evaluate --bridge residual \\",
         "    --checkpoint /kaggle/working/checkpoints/residual/best_model.pt --split val --limit 200"),
    md("### P0 + P2 — grouped split (build then smoke-train on it)"),
    code("!python scripts/phase0_build_data.py"),
    code("!python -m src.cli.train --bridge residual --split-dir data/splits --smoke \\",
         "    --output-dir /kaggle/working/ckpt_split"),
    md("### P1 — multi-tile training path (n_tiles > 1)"),
    code("!python -m src.cli.train --bridge mini_qformer --n-tiles 3 --split-dir data/splits \\",
         "    --smoke --output-dir /kaggle/working/ckpt_tiles"),
    md("### P4 — router P(r|Q): PhoBERT + 8-way reasoning-type head (contribution #2)"),
    code("!python -m src.cli.train_router --split-dir data/splits --limit 6000 --epochs 2"),
    code("import json, pathlib",
         "p = pathlib.Path('checkpoints/router/metrics.json')",
         "print(json.loads(p.read_text()) if p.exists() else 'router produced no metrics')"),
    md("### P1 — is n_tiles a real compute lever? (final-plan section 5.2)"),
    code("!python -m src.cli.profile --n-tiles 1 2 4 6 --samples 32"),
    code("import json, pathlib",
         "p = pathlib.Path('outputs/profile/pipeline_cost.json')",
         "print(json.loads(p.read_text()) if p.exists() else 'profile did not produce output')"),
    md("### FULL training run",
       "",
       "`--bridge {residual,multi_token,tile_attention,mini_qformer,qformer}`"),
    code("# whole dataset, ~hours. Uncomment when ready.",
         "# !python -m src.cli.train --bridge residual --output-dir /kaggle/working/checkpoints",
         "# resume in a later session (auto-picks newest step_*.pt):",
         "# !python -m src.cli.train --bridge residual --output-dir /kaggle/working/checkpoints --resume"),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(HERE / "kaggle_runner.ipynb").write_text(json.dumps(NB, indent=1))

META = {
    "id": "nguyennn263/modular-vlm-finetune",
    "title": "modular-vlm-finetune",
    "code_file": "kaggle_runner.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["nguynrichard/auto-vqabest"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(META, indent=2))
print("wrote kaggle_runner.ipynb + kernel-metadata.json")
