"""Build the |A|=9 oracle tables (multi_token + qformer + mini_qformer x n_tiles
1/3/6), train the 3 policy ablation arms on TRAIN, evaluate on TEST, and report
a*-match + the action histogram each policy actually picks.

    python scripts/analyze_A9.py

Expects, per split in {val, test, train}:
  outputs/oracle_<split>/_mt/table.shard*.parquet    (multi_token, |A|=3)
  outputs/oracle_<split>/_qfmq/table.shard*.parquet  (qformer+mini_qformer, |A|=6)
"""
from __future__ import annotations
import glob, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.oracle import oracle_labels          # noqa: E402
from src.reasoning_types import CATEGORIES             # noqa: E402


def build_A9(split: str) -> pd.DataFrame:
    d = ROOT / "outputs" / f"oracle_{split}"
    mt = pd.concat([pd.read_parquet(f) for f in glob.glob(str(d / "_mt" / "table.shard*.parquet"))], ignore_index=True)
    qf = pd.concat([pd.read_parquet(f) for f in glob.glob(str(d / "_qfmq" / "table.shard*.parquet"))], ignore_index=True)
    a9 = pd.concat([mt, qf], ignore_index=True).drop_duplicates(["sample_id", "action"])
    # keep only samples fully covered by all 9 actions
    full = a9.groupby("sample_id").size()
    a9 = a9[a9.sample_id.isin(full[full == 9].index)].reset_index(drop=True)
    a9.to_parquet(d / "table_A9.parquet", index=False)
    oracle_labels(a9).to_parquet(d / "labels_A9.parquet", index=False)
    print(f"[{split}] |A|=9: {a9.sample_id.nunique()} samples "
          f"(mt {mt.sample_id.nunique()}, qfmq {qf.sample_id.nunique()})")
    return a9


def train_arm(name: str, extra: list[str]) -> Path:
    out = ROOT / "checkpoints" / f"policyA9_{name}"
    cmd = [sys.executable, "-m", "src.cli.train_policy",
           "--labels", "outputs/oracle_train/labels_A9.parquet",
           "--val-labels", "outputs/oracle_test/labels_A9.parquet",
           "--val-prq", "outputs/router/prq_test.parquet",
           "--val-features", "outputs/fiq/test.parquet",
           "--epochs", "60", "--out", str(out)] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if "a*-match" in l]
    print(f"  {name:12} {line[-1] if line else r.stdout.strip()[-200:]}  {r.stderr.strip()[-120:]}")
    return out / "best.pt"


def picks_hist(ckpt: Path, lam: float = 0.2) -> dict:
    import torch
    from src.modeling.policy import PolicyMLP
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    pdim, vdim = ck.get("prq_dim", 8), ck.get("visual_dim", 0)
    m = PolicyMLP(prq_dim=pdim, visual_dim=vdim, num_actions=len(ck["actions"]))
    m.load_state_dict(ck["state_dict"]); m.eval()
    prq = pd.read_parquet("outputs/router/prq_test.parquet").drop_duplicates("sample_id")
    ids = list(prq.sample_id)
    P = torch.tensor(prq[[f"p_{c}" for c in CATEGORIES]].to_numpy(np.float32)) if pdim else None
    F = None
    if vdim:
        f = pd.read_parquet("outputs/fiq/test.parquet").drop_duplicates("sample_id").set_index("sample_id").reindex(ids)
        F = torch.tensor(f[[c for c in f.columns]].fillna(0).to_numpy(np.float32))
    with torch.no_grad():
        a = m(P, torch.full((len(ids),), lam), F).argmax(-1).numpy()
    acts = np.array(ck["actions"])[a]
    return pd.Series(acts).value_counts(normalize=True).round(3).to_dict()


def main() -> None:
    for s in ("val", "test", "train"):
        build_A9(s)
    print("\n=== policy ablation on |A|=9 (train -> test held-out) ===")
    arms = {
        "ours":        ["--prq", "outputs/router/prq_train.parquet", "--features", "outputs/fiq/train.parquet"],
        "rt_only":     ["--prq", "outputs/router/prq_train.parquet"],
        "visual_only": ["--no-prq", "--prq", "outputs/router/prq_train.parquet", "--features", "outputs/fiq/train.parquet"],
    }
    cks = {n: train_arm(n, e) for n, e in arms.items()}
    print("\n=== action histogram each policy picks (test, λ=0.2) ===")
    for n, ck in cks.items():
        if ck.exists():
            print(f"  {n:12} {picks_hist(ck)}")
    print("\n=== eval_ladder ===")
    subprocess.run([sys.executable, "-m", "src.cli.eval_ladder",
                    "--oracle", "outputs/oracle_test/table_A9.parquet",
                    "--prq", "outputs/router/prq_test.parquet", "--fiq", "outputs/fiq/test.parquet",
                    "--policies", f"ours={cks['ours']},rt_only={cks['rt_only']},visual_only={cks['visual_only']}",
                    "--out", "outputs/eval_A9"], cwd=ROOT)


if __name__ == "__main__":
    main()
