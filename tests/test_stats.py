"""Unit tests for the paired significance tests."""
import numpy as np
import pytest

from src.analysis.stats import holm, paired_bootstrap, permutation_test


def test_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.1, 500)
    b = rng.normal(0.5, 0.1, 500)
    r = paired_bootstrap(a, b)
    assert r["mean_diff"] > 0
    assert r["p_value"] < 0.05
    assert r["ci_low"] > 0


def test_bootstrap_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, 400)
    b = rng.normal(0.5, 0.1, 400)  # same distribution, independent
    r = paired_bootstrap(a, b)
    assert r["p_value"] > 0.05
    assert r["ci_low"] < 0 < r["ci_high"]


def test_permutation_agrees_with_bootstrap_sign():
    rng = np.random.default_rng(2)
    a = rng.normal(0.55, 0.1, 300)
    b = rng.normal(0.50, 0.1, 300)
    assert permutation_test(a, b)["p_value"] < 0.1


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0
    assert adj["a"] >= 0.001 * 3 - 1e-9


@pytest.mark.parametrize("fn", [paired_bootstrap, permutation_test])
def test_rejects_bad_shape(fn):
    with pytest.raises(AssertionError):
        fn(np.zeros(5), np.zeros(4))
