"""Exp B analysis on synthetic per-sample rows (no model needed)."""
import numpy as np

from src.analysis.expB import analyse
from src.reasoning_types import CATEGORIES

WEIGHTS = {c: 1 / len(CATEGORIES) for c in CATEGORIES}


def _rows(bridge, category, mean, n=120, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"image_id": i, "question": f"q{i}", "category": category,
         "bridge": bridge, "cider": float(max(0.0, rng.normal(mean, 0.15)))}
        for i in range(n)
    ]


def test_fork_detected_when_one_bridge_dominates_a_category():
    rows = []
    # 'relational': bridge B clearly better; other categories: tie
    rows += _rows("A", "relational", 0.30, seed=1)
    rows += _rows("B", "relational", 0.60, seed=2)
    rows += _rows("A", "counting", 0.50, seed=3)
    rows += _rows("B", "counting", 0.50, seed=4)
    r = analyse(rows, WEIGHTS)
    assert r["fork"]["relational"]["top"] == "B"
    assert r["fork"]["relational"]["significant"] is True
    assert r["fork"]["counting"]["significant"] is False


def test_top3_and_shapes():
    rows = []
    for b, m in [("A", 0.6), ("B", 0.5), ("C", 0.4), ("D", 0.3)]:
        for c in CATEGORIES[:4]:
            rows += _rows(b, c, m, seed=hash((b, c)) % 1000)
    r = analyse(rows, WEIGHTS)
    assert r["top3_for_action_space"] == ["A", "B", "C"]
    assert set(r["bridges"]) == {"A", "B", "C", "D"}
    assert r["verdict"]  # non-empty
