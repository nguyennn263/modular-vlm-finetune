import numpy as np
import pandas as pd

from src.analysis.oracle import action_mix, frontier, oracle_labels, utility


def _table():
    # 2 samples, 3 actions with (M, C): cheap-poor / mid / expensive-good
    rows = []
    for sid in (0, 1):
        rows += [
            {"sample_id": sid, "action": "t1", "M": 0.30, "C": 0.0},
            {"sample_id": sid, "action": "t3", "M": 0.55, "C": 0.5},
            {"sample_id": sid, "action": "t6", "M": 0.60, "C": 1.0},
        ]
    return pd.DataFrame(rows)


def test_utility_formula():
    assert utility(np.array([1.0]), np.array([0.5]), 0.4)[0] == 0.8


def test_lambda0_picks_max_quality_lambda_high_picks_cheapest():
    lab = oracle_labels(_table(), [0.0, 1.0])
    at0 = lab[lab["lambda"] == 0.0]["a_star"].unique()
    at1 = lab[lab["lambda"] == 1.0]["a_star"].unique()
    assert list(at0) == ["t6"]      # λ=0 -> best M
    assert list(at1) == ["t1"]      # λ=1 -> C dominates -> cheapest


def test_frontier_is_monotone_in_lambda():
    f = frontier(_table())
    assert f["mean_C"].is_monotonic_decreasing
    assert f["mean_M"].is_monotonic_decreasing
    assert (f["n"] == 2).all()


def test_action_mix_shares_sum_to_one():
    mix = action_mix(_table())
    s = mix.groupby("lambda")["share"].sum()
    assert np.allclose(s.to_numpy(), 1.0)
