#!/usr/bin/env python3
"""P1 - multi-tile pipeline + FLOPs/latency calibration of the n_tiles grid."""
import sys

from _phase import Step, run_phase

PY = sys.executable

STEPS = [
    Step("Profile FLOPs + latency + throughput across n_tiles",
         [PY, "-m", "src.cli.profile", "--n-tiles", "1", "2", "4", "6", "--samples", "32"]),
    Step("DONE (v8): FLOPs 6x / latency 4x / throughput 5.2x between 1 and 6 tiles "
         "-> n_tiles is the primary lever; configs/action_space.yaml frozen."),
    Step("Wire multi-tile into the training collator + trainer forward/generate", done=False),
    Step("Retrain the lever bridge with tile-count augmentation (T in 1..6)", done=False),
]

if __name__ == "__main__":
    run_phase("phase1_profile", STEPS)
