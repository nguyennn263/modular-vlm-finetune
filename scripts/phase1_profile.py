#!/usr/bin/env python3
"""P1 - multi-tile pipeline + FLOPs/latency calibration of the n_tiles grid."""
import sys

from _phase import Step, run_phase

PY = sys.executable

STEPS = [
    Step("Implement multi-tile forward/generate (B,T,3,H,W) -> InternViT -> regroup", done=False),
    Step("Train the tile-capable bridge with tile-count augmentation (T in 1..6)", done=False),
    Step("Profile FLOPs + latency + throughput across n_tiles",
         [PY, "-m", "src.cli.profile", "--n-tiles", "1", "2", "3", "4", "6", "--samples", "200"],
         done=False),
    Step("Freeze n_tiles grid + C(a) coefficients in configs/action_space.yaml", done=False),
]

if __name__ == "__main__":
    run_phase("phase1_profile", STEPS)
