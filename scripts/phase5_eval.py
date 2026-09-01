#!/usr/bin/env python3
"""P5 - full evaluation on TEST: ablation ladder, Pareto, compute table, human eval."""
from _phase import Step, run_phase

STEPS = [
    Step("Ablation ladder (7 arms) on TEST, >=3 seeds", done=False),
    Step("Pareto frontier: M vs C and M vs latency, per arm", done=False),
    Step("Compute-efficiency table: FLOPs / latency / throughput / trainable-params", done=False),
    Step("Policy Behavior Analysis: action distribution by category and reason_depth", done=False),
    Step("Human validation (300-500 QA) + quantitative error analysis", done=False),
]

if __name__ == "__main__":
    run_phase("phase5_eval", STEPS)
