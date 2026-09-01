#!/usr/bin/env python3
"""P2 - Exp A: train the 6 bridge baselines on the official split (3 seeds each)."""
import sys

from _phase import Step, run_phase

PY = sys.executable
BRIDGES = ["residual", "multi_token", "tile_attention", "mini_qformer", "qformer", "gated_fusion"]

STEPS = [
    Step(f"Train {b} (n_tiles=1, seeds 42/43/44 via --seed)",
         [PY, "-m", "src.cli.train", "--bridge", b])
    for b in BRIDGES
] + [
    Step("Evaluate every checkpoint on val + aggregate mean/std", done=False),
]

if __name__ == "__main__":
    run_phase("phase2_expA", STEPS)
