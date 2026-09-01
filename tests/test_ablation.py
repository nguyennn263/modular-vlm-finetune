import numpy as np
import pandas as pd

from src.analysis.ablation import ladder, pareto_front


def _table(n=60):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        # t1 cheap/ok, t3 mid, t6 expensive/best — with noise
        rows += [
            {"sample_id": i, "action": "t1", "M": rng.normal(0.35, 0.05), "C": 0.0},
            {"sample_id": i, "action": "t3", "M": rng.normal(0.50, 0.05), "C": 0.5},
            {"sample_id": i, "action": "t6", "M": rng.normal(0.58, 0.05), "C": 1.0},
        ]
    return pd.DataFrame(rows)


def test_ladder_has_all_arms_and_oracle_dominates():
    l = ladder(_table())
    arms = set(l["arm"])
    assert {"fixed:t1", "fixed:t3", "fixed:t6", "random", "oracle"} <= arms

    # oracle at λ=0 should reach ~the best fixed M, at lower or equal C on average
    orc0 = l[(l.arm == "oracle") & (l["lambda"] == 0.0)].iloc[0]
    best_fixed_M = l[l.arm.str.startswith("fixed:")]["mean_M"].max()
    assert orc0["mean_M"] >= best_fixed_M - 0.02

    # oracle C decreases as λ grows
    oc = l[l.arm == "oracle"].sort_values("lambda")["mean_C"].to_numpy()
    assert oc[0] >= oc[-1]


def test_trained_policy_picks_are_used():
    tbl = _table(30)
    picks = {lam: {i: "t3" for i in range(30)} for lam in [0.0, 1.0]}
    l = ladder(tbl, policy_picks={"ours": picks}, lambdas=[0.0, 1.0])
    ours = l[l.arm == "ours"]
    assert len(ours) == 2
    assert np.allclose(ours["mean_C"].to_numpy(), 0.5)


def test_pareto_is_non_dominated():
    l = ladder(_table())
    pf = pareto_front(l)
    assert pf["mean_C"].is_monotonic_increasing
    assert pf["mean_M"].is_monotonic_increasing
