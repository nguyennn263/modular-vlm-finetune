#!/usr/bin/env python3
"""P2 - Exp A: 5 bridge baselines x 3 seeds on the official grouped split.

This is a multi-hour job (15 training runs). Launch deliberately; each run
checkpoints and can be resumed with `--resume`.
"""
import sys

from _phase import Step, run_phase

PY = sys.executable
BRIDGES = ["residual", "multi_token", "tile_attention", "mini_qformer", "qformer"]
SEEDS = [42, 43, 44]

STEPS = [Step("Build the grouped split (idempotent)",
              [PY, "scripts/phase0_build_data.py"])]
STEPS += [
    Step(f"Train {b} seed {s}",
         [PY, "-m", "src.cli.train", "--bridge", b, "--split-dir", "data/splits",
          "--seed", str(s), "--output-dir", f"checkpoints/expA/seed{s}", "--resume"])
    for b in BRIDGES for s in SEEDS
]
STEPS.append(Step("Evaluate every checkpoint on val + aggregate mean/std -> outputs/expA/", done=False))

if __name__ == "__main__":
    run_phase("phase2_expA", STEPS)
