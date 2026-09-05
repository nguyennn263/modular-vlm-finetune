"""C3: re-lock the |A|=9 oracle analysis (§5.2/§5.3) on TILE-TRAINED checkpoints.

Unlike the original 1-tile-checkpoint sweep, this one runs all 3 bridges x {1,3,6}
tiles in a single oracle worker per shard, so `outputs/oracle_{val,test}_tiled/`
already holds full |A|=9 tables per shard — no _mt/_qfmq merge needed.

Policy is still trained on the ORIGINAL (1-tile-checkpoint) train labels
(`outputs/oracle_train/labels_A9.parquet`) — train was not re-swept (see
plans/P6-draft/05-results.md Pending: this can only bias against a learned
policy, not manufacture a false positive).

    python scripts/analyze_A9_tiled.py
"""
from __future__ import annotations
import glob, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.oracle import oracle_labels          # noqa: E402
from src.reasoning_types import CATEGORIES              # noqa: E402


def build(split: str) -> pd.DataFrame:
    d = ROOT / "outputs" / f"oracle_{split}_tiled"
    a9 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "table.shard*.parquet")))],
                   ignore_index=True).drop_duplicates(["sample_id", "action"])
    full = a9.groupby("sample_id").size()
    a9 = a9[a9.sample_id.isin(full[full == 9].index)].reset_index(drop=True)
    a9.to_parquet(d / "table_A9.parquet", index=False)
    oracle_labels(a9).to_parquet(d / "labels_A9.parquet", index=False)
    print(f"[{split}-tiled] |A|=9: {a9.sample_id.nunique()} samples, "
          f"actions={sorted(a9.action.unique())}")
    return a9


def headroom_table(a9: pd.DataFrame, label: str) -> None:
    """§5.2-style summary: oracle vs fixed:multi_token|t1 vs random, mean CIDEr/cost."""
    from src.analysis.oracle import utility
    m = a9.set_index(["sample_id", "action"])["M"]
    c = a9.set_index(["sample_id", "action"])["C"]

    def fixed(action):
        sub = a9[a9.action == action]
        return sub.M.mean(), sub.C.mean()

    lab0 = oracle_labels(a9, lambdas=[0.0])
    oracle_m, oracle_c = lab0.M_star.mean(), lab0.C_star.mean()
    fx_m, fx_c = fixed("multi_token|t1")
    rand_m = a9.groupby("sample_id").M.apply(lambda s: s.sample(1, random_state=0).iloc[0]).mean()
    print(f"\n=== {label}: oracle vs fixed vs random ===")
    print(f"  oracle a*(x,0):        M={oracle_m:.3f}  C={oracle_c:.3f}")
    print(f"  fixed multi_token|t1:  M={fx_m:.3f}  C={fx_c:.3f}")
    print(f"  random:                M={rand_m:.3f}")
    for br in ("multi_token", "qformer", "mini_qformer"):
        for t in (1, 3, 6):
            a = f"{br}|t{t}"
            if a in a9.action.values:
                sub = a9[a9.action == a]
                print(f"  {a:20} M={sub.M.mean():.3f}")


def train_arm(name: str, extra: list[str]) -> Path:
    out = ROOT / "checkpoints" / f"policyA9_tiled_{name}"
    cmd = [sys.executable, "-m", "src.cli.train_policy",
           "--labels", "outputs/oracle_train/labels_A9.parquet",      # NOT re-swept (see docstring)
           "--val-labels", "outputs/oracle_test_tiled/labels_A9.parquet",
           "--val-prq", "outputs/router/prq_test.parquet",
           "--epochs", "60", "--out", str(out)] + extra
    if "--features" in extra:
        cmd += ["--val-features", "outputs/fiq/test.parquet"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if "a*-match" in l]
    print(f"  {name:12} {line[-1] if line else r.stdout.strip()[-200:]}  {r.stderr.strip()[-150:]}")
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
    a9_val = build("val")
    a9_test = build("test")
    headroom_table(a9_val, "val-tiled")
    headroom_table(a9_test, "test-tiled")

    print("\n=== policy ablation on |A|=9 TILED (1-tile-ckpt TRAIN -> tiled-ckpt TEST held-out) ===")
    arms = {
        "ours":        ["--prq", "outputs/router/prq_train.parquet", "--features", "outputs/fiq/train.parquet"],
        "rt_only":     ["--prq", "outputs/router/prq_train.parquet"],
        "visual_only": ["--no-prq", "--prq", "outputs/router/prq_train.parquet", "--features", "outputs/fiq/train.parquet"],
    }
    cks = {n: train_arm(n, e) for n, e in arms.items()}
    print("\n=== action histogram each policy picks (test-tiled, λ=0.2) ===")
    for n, ck in cks.items():
        if ck.exists():
            print(f"  {n:12} {picks_hist(ck)}")

    print("\n=== eval_ladder (tiled) ===")
    subprocess.run([sys.executable, "-m", "src.cli.eval_ladder",
                    "--oracle", "outputs/oracle_test_tiled/table_A9.parquet",
                    "--prq", "outputs/router/prq_test.parquet", "--fiq", "outputs/fiq/test.parquet",
                    "--policies", f"ours={cks['ours']},rt_only={cks['rt_only']},visual_only={cks['visual_only']}",
                    "--out", "outputs/eval_A9_tiled"], cwd=ROOT)


if __name__ == "__main__":
    main()
