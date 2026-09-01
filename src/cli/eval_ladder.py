"""P5 — ablation ladder + Pareto + policy-behaviour on TEST.

    python -m src.cli.eval_ladder \
        --oracle outputs/oracle_test/table.parquet \
        --prq outputs/router/prq_test.parquet --fiq outputs/fiq/test.parquet \
        --policies ours=checkpoints/policy_ours/best.pt,rt_only=checkpoints/policy_rt/best.pt

Every arm's (mean M, mean C) at each λ -> outputs/eval/{ladder.csv, pareto.csv,
behaviour.json}. fixed:* / random / oracle need no checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.oracle import LAMBDA_GRID
from src.config.loader import repo_root


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.eval_ladder", description=__doc__)
    p.add_argument("--oracle", default="outputs/oracle_test/table.parquet")
    p.add_argument("--prq", default="outputs/router/prq_test.parquet")
    p.add_argument("--fiq", default=None)
    p.add_argument("--policies", default="", help="name=ckpt,name=ckpt (trained policy arms)")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--dry-run", action="store_true")
    return p


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else repo_root() / p


def _policy_picks(ckpt: str, prq_df, fiq_df, actions_ref):
    import numpy as np
    import torch

    from src.modeling.policy import PolicyMLP
    from src.reasoning_types import CATEGORIES

    ck = torch.load(_abs(ckpt), map_location="cpu", weights_only=False)
    actions = ck["actions"]
    pdim, vdim = ck.get("prq_dim", len(CATEGORIES)), ck.get("visual_dim", 0)
    model = PolicyMLP(prq_dim=pdim, visual_dim=vdim, num_actions=len(actions))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ids = list(prq_df["sample_id"])
    prq = torch.tensor(prq_df[[f"p_{c}" for c in CATEGORIES]].to_numpy(np.float32)) if pdim else None
    fiq = None
    if vdim:
        f = fiq_df.set_index("sample_id").loc[ids]
        fiq = torch.tensor(f[[c for c in f.columns]].to_numpy(np.float32))

    picks = {}
    for lam in LAMBDA_GRID:
        lt = torch.full((len(ids),), lam)
        with torch.no_grad():
            a = model(prq, lt, fiq).argmax(-1).tolist()
        picks[lam] = {sid: actions[i] for sid, i in zip(ids, a)}
    return picks


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    import pandas as pd

    from src.analysis.ablation import ladder, pareto_front

    table = pd.read_parquet(_abs(args.oracle))
    print(f"[ladder] TEST oracle table: {table.shape}, "
          f"{table.sample_id.nunique()} samples x {table.action.nunique()} actions")
    if args.dry_run:
        return

    prq_df = pd.read_parquet(_abs(args.prq))
    fiq_df = pd.read_parquet(_abs(args.fiq)) if args.fiq else None

    policy_picks = {}
    for spec in [s for s in args.policies.split(",") if "=" in s]:
        name, ckpt = spec.split("=", 1)
        policy_picks[name.strip()] = _policy_picks(ckpt.strip(), prq_df, fiq_df, table)
        print(f"[ladder] loaded policy arm '{name.strip()}'")

    lad = ladder(table, policy_picks=policy_picks)
    pf = pareto_front(lad)

    out = repo_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    lad.to_csv(out / "ladder.csv", index=False)
    pf.to_csv(out / "pareto.csv", index=False)

    # policy-behaviour: action distribution by category (for the 'ours' arm, λ~0.2)
    behaviour = {}
    if "ours" in policy_picks:
        from src.data.split import load_split
        cat = {f"{(s.metadata or {}).get('image_id')}::{s.question}": s.metadata["category"]
               for s in load_split("test", args.split_dir)}
        picks = policy_picks["ours"].get(0.2, next(iter(policy_picks["ours"].values())))
        by_cat: dict = {}
        for sid, a in picks.items():
            by_cat.setdefault(cat.get(sid, "?"), {}).setdefault(a, 0)
            by_cat[cat.get(sid, "?")][a] += 1
        behaviour = by_cat
    (out / "behaviour.json").write_text(json.dumps(behaviour, indent=2, ensure_ascii=False))

    print(lad.to_string(index=False))
    print(f"\n[pareto]\n{pf.to_string(index=False)}\n[ladder] -> {out}/")


if __name__ == "__main__":
    main()
