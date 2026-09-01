#!/usr/bin/env python3
"""P4 - oracle utility-cost sweep, router P(r|Q)/f(I,Q), and the policy network."""
from _phase import Step, run_phase

STEPS = [
    Step("Oracle sweep: 7.5k TRAIN subset x 9 actions, greedy, 1 seed -> M(a),C(a)", done=False),
    Step("Generate a*(x, lambda) for the 7-point lambda grid", done=False),
    Step("Train router: PrQHead (PhoBERT) + VisualStateProbe", done=False),
    Step("Train PolicyMLP on (P(r|Q), f(I,Q), lambda) -> a* with cross-entropy", done=False),
    Step("Pick operating lambda on val -> configs/lambda_operating.yaml", done=False),
]

if __name__ == "__main__":
    run_phase("phase4_oracle_policy", STEPS)
