import numpy as np
import pandas as pd
import pytest

from src.cli.train_policy import _assemble
from src.reasoning_types import CATEGORIES


def _write(tmp_path):
    ids = [f"img{i}::q{i}" for i in range(40)]
    lab = pd.DataFrame({"sample_id": ids * 2,
                        "lambda": [0.0] * 40 + [1.0] * 40,
                        "a_star": (["mini_qformer|t6"] * 40) + (["residual|t1"] * 40)})
    prq = pd.DataFrame({"sample_id": ids,
                        **{f"p_{c}": np.random.rand(40) for c in CATEGORIES}})
    fiq = pd.DataFrame({"sample_id": ids, **{f"f{j}": np.random.rand(40) for j in range(8)}})
    lab.to_parquet(tmp_path / "labels.parquet")
    prq.to_parquet(tmp_path / "prq.parquet")
    fiq.to_parquet(tmp_path / "fiq.parquet")


def test_assemble_without_features(tmp_path):
    _write(tmp_path)
    X_prq, X_lam, X_fiq, y, actions = _assemble(
        str(tmp_path / "labels.parquet"), str(tmp_path / "prq.parquet"), None)
    assert X_prq.shape == (80, 8)
    assert X_lam.shape == (80,)
    assert X_fiq is None
    assert set(actions) == {"mini_qformer|t6", "residual|t1"}
    assert len(y) == 80


def test_assemble_with_features(tmp_path):
    _write(tmp_path)
    *_, X_fiq, y, _ = _assemble(
        str(tmp_path / "labels.parquet"), str(tmp_path / "prq.parquet"),
        str(tmp_path / "fiq.parquet"))
    assert X_fiq.shape == (80, 8)


def test_policy_mlp_forward_shapes():
    torch = pytest.importorskip("torch")
    from src.modeling.policy import PolicyMLP

    m = PolicyMLP(prq_dim=8, visual_dim=0, num_actions=9)
    out = m(torch.randn(4, 8), torch.tensor([0.0, 0.5, 1.0, 0.2]))
    assert out.shape == (4, 9)

    m2 = PolicyMLP(prq_dim=8, visual_dim=16, num_actions=9)
    out2 = m2(torch.randn(4, 8), torch.zeros(4), torch.randn(4, 16))
    assert out2.shape == (4, 9)
