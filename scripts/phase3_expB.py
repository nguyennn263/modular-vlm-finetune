#!/usr/bin/env python3
"""P3 - Exp B: bridge x category heatmap + paired-bootstrap fork decision."""
from _phase import Step, run_phase

STEPS = [
    Step("Join bridge val predictions with category -> per-cell CIDEr/Acc/F1 heatmap", done=False),
    Step("Paired bootstrap + permutation test: best vs 2nd best per category", done=False),
    Step("Apply the 4 gates; pick top-3 bridges -> configs/action_space.yaml:bridges", done=False),
]

if __name__ == "__main__":
    run_phase("phase3_expB", STEPS)
