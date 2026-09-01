#!/usr/bin/env python3
"""P5 - full evaluation on TEST: ablation ladder, Pareto, compute table, human eval.

Prereqs: Exp A checkpoints (P2), router (P4), the trained policy arms.
"""
import sys

from _phase import Step, run_phase

PY = sys.executable
POLICY_ARMS = {
    "ours": ["--features", "outputs/fiq/train.parquet"],
    "rt_only": [],
    "visual_only": ["--no-prq", "--features", "outputs/fiq/train.parquet"],
}

STEPS = [
    Step("Predict P(r|Q) + f(I,Q) for TEST",
         [PY, "-m", "src.cli.train_router", "--from-checkpoint", "checkpoints/router/best.pt",
          "--predict-splits", "test"]),
    Step("Build f(I,Q) for TEST", [PY, "-m", "src.cli.build_fiq", "--splits", "test"]),
    Step("Oracle sweep on TEST (all 9 actions)",
         [PY, "-m", "src.cli.oracle", "--split", "test", "--subset", "100000",
          "--bridges", "mini_qformer,qformer,residual", "--n-tiles", "1,3,6",
          "--ckpt-dir", "checkpoints/expA/seed42", "--out", "outputs/oracle_test"]),
]
STEPS += [
    Step(f"Train policy arm '{arm}'",
         [PY, "-m", "src.cli.train_policy", "--out", f"checkpoints/policy_{arm}",
          "--prq", "outputs/router/prq_train.parquet", "--labels", "outputs/oracle/labels.parquet",
          "--val-prq", "outputs/router/prq_val.parquet", "--val-labels", "outputs/oracle_val/labels.parquet",
          *extra])
    for arm, extra in POLICY_ARMS.items()
]
STEPS += [
    Step("Ablation ladder + Pareto + policy behaviour on TEST",
         [PY, "-m", "src.cli.eval_ladder", "--fiq", "outputs/fiq/test.parquet",
          "--policies", "ours=checkpoints/policy_ours/best.pt,"
                        "rt_only=checkpoints/policy_rt_only/best.pt,"
                        "visual_only=checkpoints/policy_visual_only/best.pt"]),
    Step("Compute-efficiency table (FLOPs/latency/throughput/params) from profile + ckpt sizes", done=False),
    Step("Human validation (300-500 QA) + quantitative error analysis", done=False),
]

if __name__ == "__main__":
    run_phase("phase5_eval", STEPS)
