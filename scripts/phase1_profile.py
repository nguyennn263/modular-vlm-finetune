#!/usr/bin/env python3
"""P1 - multi-tile pipeline + FLOPs/latency calibration of the n_tiles grid."""
import sys

from _phase import Step, run_phase

PY = sys.executable

STEPS = [
    Step("Profile FLOPs + latency + throughput across n_tiles (tile_attention bridge)",
         [PY, "-m", "src.cli.profile", "--n-tiles", "1", "2", "4", "6", "--samples", "64"]),
    Step("Read outputs/profile/pipeline_cost.json -> decide the lever (final-plan 5.2)", done=False),
    Step("Freeze n_tiles grid + C(a) coefficients in configs/action_space.yaml", done=False),
    Step("Wire multi-tile into the training collator + trainer forward/generate", done=False),
    Step("Retrain the lever bridge with tile-count augmentation (T in 1..6)", done=False),
]

if __name__ == "__main__":
    run_phase("phase1_profile", STEPS)
