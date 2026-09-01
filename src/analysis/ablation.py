"""Ablation ladder + Pareto frontier on TEST (final-plan P5).

Every arm is a *policy* = a map (sample_id, λ) -> action. Given the TEST oracle
table ``[sample_id, action, M, C]`` (all 9 actions scored per sample), each arm's
(mean M, mean C) at each λ is a point; the empirical frontier over arms/λ is the
Pareto plot.

Arms:
  fixed:<action>        - always that action (λ-independent)
  random               - uniform over actions per sample (seeded)
  oracle               - a*(x, λ) with known M (upper bound)
  <name>               - a trained policy: pass its picks in `policy_picks`
                         (reasoning_type_only / visual_state_only / ours /
                          oracle_cognitive_prior)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.oracle import LAMBDA_GRID, oracle_labels


def _mc_lookup(table: pd.DataFrame):
    m = {(r.sample_id, r.action): r.M for r in table.itertuples()}
    c = {a: table.loc[table.action == a, "C"].iloc[0] for a in table.action.unique()}
    return m, c


def ladder(table: pd.DataFrame,
           policy_picks: dict[str, dict] | None = None,
           lambdas: list[float] = LAMBDA_GRID,
           seed: int = 42) -> pd.DataFrame:
    """Returns long-form [arm, lambda, mean_M, mean_C, n]."""
    for col in ("sample_id", "action", "M", "C"):
        if col not in table.columns:
            raise ValueError(f"table missing {col!r}")
    M, C = _mc_lookup(table)
    samples = list(table.sample_id.unique())
    actions = sorted(table.action.unique())
    rows = []

    def add(arm, lam, picks: dict):
        ms = [M[(s, picks[s])] for s in samples if (s, picks[s]) in M]
        cs = [C[picks[s]] for s in samples]
        rows.append({"arm": arm, "lambda": lam, "mean_M": float(np.mean(ms)),
                     "mean_C": float(np.mean(cs)), "n": len(ms)})

    # fixed-budget sweep (one constant action)
    for a in actions:
        add(f"fixed:{a}", None, {s: a for s in samples})

    # random
    rng = np.random.default_rng(seed)
    rand_pick = {s: actions[int(rng.integers(len(actions)))] for s in samples}
    for lam in lambdas:
        add("random", lam, rand_pick)

    # oracle upper bound
    lab = oracle_labels(table, lambdas)
    for lam in lambdas:
        picks = dict(zip(lab.loc[lab["lambda"] == lam, "sample_id"],
                         lab.loc[lab["lambda"] == lam, "a_star"]))
        add("oracle", lam, {s: picks.get(s, actions[0]) for s in samples})

    # trained policies
    for arm, per_lambda in (policy_picks or {}).items():
        for lam in lambdas:
            picks = per_lambda.get(lam) or per_lambda.get(round(lam, 4))
            if picks:
                add(arm, lam, {s: picks.get(s, actions[0]) for s in samples})

    return pd.DataFrame(rows)


def pareto_front(ladder_df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated (min C, max M) points across all arms/λ."""
    pts = ladder_df.sort_values(["mean_C", "mean_M"], ascending=[True, False]).reset_index(drop=True)
    keep, best_m = [], -np.inf
    for _, r in pts.iterrows():
        if r["mean_M"] > best_m:
            keep.append(r)
            best_m = r["mean_M"]
    return pd.DataFrame(keep)
