#!/usr/bin/env python3
"""P4 - oracle utility-cost sweep, router P(r|Q)/f(I,Q), and the policy network."""
import sys

from _phase import Step, run_phase

PY = sys.executable

STEPS = [
    Step("Train router P(r|Q) + dump prq_{train,val}.parquet (contribution #2)",
         [PY, "-m", "src.cli.train_router", "--split-dir", "data/splits"]),
    Step("Oracle sweep: TRAIN subset x (n_tiles x top-3 bridges), greedy -> table.parquet + labels.parquet",
         [PY, "-m", "src.cli.oracle", "--bridges", "mini_qformer,qformer,residual",
          "--n-tiles", "1,3,6", "--subset", "7500", "--ckpt-dir", "checkpoints/expA/seed42"]),
    Step("Build f(I,Q) visual-state features (pooled InternViT @ n_tiles=1 + metadata)", done=False),
    Step("Train PolicyMLP on (P(r|Q), [f(I,Q),] lambda) -> a*",
         [PY, "-m", "src.cli.train_policy", "--prq", "outputs/router/prq_train.parquet",
          "--labels", "outputs/oracle/labels.parquet"]),
    Step("Pick operating lambda on val -> configs/lambda_operating.yaml", done=False),
]

if __name__ == "__main__":
    run_phase("phase4_oracle_policy", STEPS)
