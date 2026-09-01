"""Oracle utility-cost logic (final-plan section 5.4).

    U(a; x, λ) = M(a; x) − λ · C(a)
    a*(x, λ)   = argmax_a U(a; x, λ)

Pure functions over a long-form table with columns
``[sample_id, action, M, C]`` — the heavy generation that fills that table lives
in ``src.cli.oracle``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


def utility(m: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    return np.asarray(m, float) - lam * np.asarray(c, float)


def oracle_labels(table: pd.DataFrame, lambdas: list[float] = LAMBDA_GRID) -> pd.DataFrame:
    """table: long-form [sample_id, action, M, C]. Returns [sample_id, lambda, a_star, U_star]."""
    for col in ("sample_id", "action", "M", "C"):
        if col not in table.columns:
            raise ValueError(f"table missing column {col!r}")
    out = []
    for lam in lambdas:
        t = table.assign(U=utility(table["M"].to_numpy(), table["C"].to_numpy(), lam))
        best = t.loc[t.groupby("sample_id")["U"].idxmax()]
        for _, r in best.iterrows():
            out.append({"sample_id": r["sample_id"], "lambda": lam,
                        "a_star": r["action"], "U_star": float(r["U"]),
                        "M_star": float(r["M"]), "C_star": float(r["C"])})
    return pd.DataFrame(out)


def frontier(table: pd.DataFrame, lambdas: list[float] = LAMBDA_GRID) -> pd.DataFrame:
    """Mean (M, C) of the oracle policy at each λ — the empirical Pareto points."""
    lab = oracle_labels(table, lambdas)
    g = lab.groupby("lambda").agg(mean_M=("M_star", "mean"), mean_C=("C_star", "mean"),
                                  n=("sample_id", "count")).reset_index()
    return g


def action_mix(table: pd.DataFrame, lambdas: list[float] = LAMBDA_GRID) -> pd.DataFrame:
    """How the oracle action distribution shifts with λ (sanity / behaviour analysis)."""
    lab = oracle_labels(table, lambdas)
    counts = lab.groupby(["lambda", "a_star"]).size().rename("count").reset_index()
    totals = counts.groupby("lambda")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts[["lambda", "a_star", "share"]]
