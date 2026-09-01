"""Paired significance tests (final-plan section 9).

Both operate on per-sample scores for two systems on the *same* items:
- ``paired_bootstrap``: resample items with replacement, CI + P(A>B).
- ``permutation_test``: shuffle which system each item's pair belongs to.

The decision rule in the plan (Exp B fork, P5 comparisons) is the paired
bootstrap / permutation p-value — nothing else.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 42,
                     ci: float = 0.95) -> dict:
    """a, b: per-sample scores for systems A and B on the same items."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    obs = float(a.mean() - b.mean())
    # two-sided p: fraction of resamples on the opposite side of 0 from obs, x2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if obs != 0 else 1.0
    return {
        "mean_diff": obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "prob_a_gt_b": float((diffs > 0).mean()),
        "n_items": int(len(a)),
    }


def permutation_test(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 42) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 1
    rng = np.random.default_rng(seed)
    obs = float(a.mean() - b.mean())
    d = a - b
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = (signs * d).mean(1)
    p = (np.abs(perm) >= abs(obs)).mean()
    return {"mean_diff": obs, "p_value": float(p), "n_items": int(len(a))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests (e.g. per category)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
