import torch

from src.training.setup import pool_for_pooled_bridge


def _hidden(b=2, t=3, p=5, d=4):
    return torch.randn(b, t * p, d), t, p


def test_default_single_tile_is_cls():
    h, _, _ = _hidden(t=1)
    assert torch.equal(pool_for_pooled_bridge(h, 1), h[:, 0])


def test_default_multi_tile_is_mean_over_all_tokens():
    h, t, _ = _hidden()
    assert torch.allclose(pool_for_pooled_bridge(h, t), h.mean(dim=1))


def test_mean_all_single_tile():
    h, _, _ = _hidden(t=1)
    assert torch.allclose(pool_for_pooled_bridge(h, 1, "mean_all"), h.mean(dim=1))


def test_cls_mean_averages_each_tiles_cls():
    h, t, p = _hidden()
    expected = torch.stack([h[:, i * p] for i in range(t)], dim=1).mean(dim=1)
    assert torch.allclose(pool_for_pooled_bridge(h, t, "cls_mean"), expected)
