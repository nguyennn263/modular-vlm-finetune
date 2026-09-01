"""Train the policy network on oracle labels (final-plan P4).

    python -m src.cli.train_policy --prq outputs/router/prq_train.parquet \
        --labels outputs/oracle/labels.parquet [--features outputs/fiq/train.parquet]

Inputs (all keyed by sample_id):
- --labels   : oracle a*(x, λ)   [sample_id, lambda, a_star]   (src.cli.oracle)
- --prq      : P(r|Q) 8-vector   [sample_id, p_<cat>...]        (src.cli.train_router --predict)
- --features : f(I,Q) d-vector   [sample_id, f0..f{d-1}]        (src.cli.build_fiq)  -- optional

Trains one policy conditioned on λ. Omitting --features => the Reasoning-type-only
ablation arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.loader import repo_root
from src.reasoning_types import CATEGORIES


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.train_policy", description=__doc__)
    p.add_argument("--labels", default="outputs/oracle/labels.parquet")
    p.add_argument("--prq", default="outputs/router/prq_train.parquet")
    p.add_argument("--no-prq", action="store_true", dest="no_prq",
                   help="Drop P(r|Q) -> visual-state-only ablation arm.")
    p.add_argument("--features", default=None, help="f(I,Q) parquet; omit for reasoning-type-only")
    p.add_argument("--val-labels", default="outputs/oracle/labels_val.parquet", dest="val_labels")
    p.add_argument("--val-prq", default="outputs/router/prq_val.parquet", dest="val_prq")
    p.add_argument("--val-features", default=None, dest="val_features")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out", default="checkpoints/policy")
    p.add_argument("--dry-run", action="store_true")
    return p


def _assemble(labels_p, prq_p, feat_p, actions: list | None = None):
    """actions: pass the TRAIN action list so val labels map to the same index space."""
    import numpy as np
    import pandas as pd

    def rp(p):
        return pd.read_parquet(p if Path(p).is_absolute() else repo_root() / p)

    lab = rp(labels_p)
    prq = rp(prq_p).drop_duplicates("sample_id")
    df = lab.merge(prq, on="sample_id", how="inner")
    feat_cols = []
    if feat_p:
        f = rp(feat_p).drop_duplicates("sample_id")
        feat_cols = [c for c in f.columns if c != "sample_id"]
        df = df.merge(f, on="sample_id", how="inner")

    if actions is None:
        actions = sorted(df["a_star"].unique())
    a2i = {a: i for i, a in enumerate(actions)}
    df = df[df["a_star"].isin(a2i)]  # drop val rows whose a* never appeared in train
    prq_cols = [f"p_{c}" for c in CATEGORIES]
    X_prq = df[prq_cols].to_numpy(np.float32)
    X_lam = df["lambda"].to_numpy(np.float32)
    X_fiq = df[feat_cols].to_numpy(np.float32) if feat_cols else None
    y = df["a_star"].map(a2i).to_numpy()
    return X_prq, X_lam, X_fiq, y, actions


def run(args: argparse.Namespace) -> dict:
    X_prq, X_lam, X_fiq, y, actions = _assemble(args.labels, args.prq, args.features)
    print(f"[policy] {len(y)} (sample,λ) rows | actions={actions} | "
          f"f(I,Q)={'on' if X_fiq is not None else 'off'}")
    if args.dry_run:
        return {"actions": actions, "n": int(len(y))}

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.modeling.policy import PolicyMLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vdim = X_fiq.shape[1] if X_fiq is not None else 0
    pdim = 0 if args.no_prq else len(CATEGORIES)
    if args.no_prq:
        X_prq = np.zeros((len(y), 0), np.float32)

    def loader(Xp, Xl, Xf, yy, shuffle):
        tensors = [torch.tensor(Xp), torch.tensor(Xl)]
        tensors.append(torch.tensor(Xf) if Xf is not None else torch.zeros(len(yy), 0))
        tensors.append(torch.tensor(yy))
        return DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=shuffle)

    train_dl = loader(X_prq, X_lam, X_fiq, y, True)
    try:
        vp, vl, vf, vy, _ = _assemble(args.val_labels, args.val_prq, args.val_features, actions=actions)
        if args.no_prq:
            vp = np.zeros((len(vy), 0), np.float32)
        val_dl = loader(vp, vl, vf, vy, False)
    except Exception as exc:  # noqa: BLE001
        print(f"[policy] no val set ({exc!r}); reporting train accuracy only")
        val_dl = None

    model = PolicyMLP(prq_dim=pdim, visual_dim=vdim,
                      num_actions=len(actions), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()

    def _split(prq, fiq):
        return (prq.to(device) if pdim else None, fiq.to(device) if vdim else None)

    def evaluate(dl):
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for prq, lam, fiq, yy in dl:
                p, f = _split(prq, fiq)
                logits = model(p, lam.to(device), f)
                ok += (logits.argmax(-1).cpu() == yy).sum().item()
                tot += len(yy)
        return ok / tot

    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for prq, lam, fiq, yy in train_dl:
            opt.zero_grad()
            p, f = _split(prq, fiq)
            loss = lossf(model(p, lam.to(device), f), yy.to(device))
            loss.backward()
            opt.step()
        acc = evaluate(val_dl) if val_dl is not None else evaluate(train_dl)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "actions": actions,
                        "prq_dim": pdim, "visual_dim": vdim}, out / "best.pt")
    report = {"best_action_accuracy_vs_oracle": round(best, 4), "actions": actions,
              "arm": ("visual_state_only" if args.no_prq else
                      "reasoning_type_only" if X_fiq is None else "ours"),
              "n_rows": int(len(y)), "prq": not args.no_prq, "fiq": X_fiq is not None}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"[policy] best a*-match = {best:.4f} -> {out}")
    return report


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
