#!/usr/bin/env python3
"""P3 - Exp B: bridge x category + paired-bootstrap fork decision.

Prerequisite: Exp A checkpoints (phase2) + per-sample eval files. Run per bridge:
    python -m src.cli.evaluate --bridge <b> --split-dir data/splits --split val \
        --checkpoint checkpoints/expA/seed42/<b>/best_model.pt
then this phase aggregates them.
"""
import sys

from _phase import Step, run_phase

PY = sys.executable
BRIDGES = ["residual", "multi_token", "tile_attention", "mini_qformer", "qformer"]

STEPS = [
    Step(f"Per-sample val eval — {b} (writes eval_val_samples.jsonl)",
         [PY, "-m", "src.cli.evaluate", "--bridge", b, "--split-dir", "data/splits",
          "--split", "val",
          "--checkpoint", f"checkpoints/expA/seed42/{b}/best_model.pt"])
    for b in BRIDGES
] + [
    Step("Fork analysis: heatmap + paired bootstrap + top-3 bridges",
         [PY, "-m", "src.analysis.expB",
          "--glob", "checkpoints/expA/**/eval_val_samples.jsonl"]),
    Step("Write top-3 bridges into configs/action_space.yaml:bridges", done=False),
]

if __name__ == "__main__":
    run_phase("phase3_expB", STEPS)
