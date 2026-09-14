"""Shape / forward-pass / gradient-flow tests for every bridge module.

Imports bridge classes directly from `src.modeling.bridge_modules` -- bypasses
`create_finetune_model`/`BridgeTrainer` entirely (those require downloading and
loading the full ~1B-param Vintern-1B HF model, which is what the existing
heavy `tests/checks/*.py` scripts do; this file is meant to be the FAST,
CPU-only, no-download pre-flight gate run before any Kaggle GPU-hours are
spent on a new bridge architecture).

Before this file, there was no shape/forward-pass test for any bridge module
at all -- `tests/checks/validate_vintern_integration.py::check_bridge_compatibility()`
only asserts `hasattr(bridge, 'bridge')`, never calls `.forward()`.
"""
import pytest
import torch

from src.modeling.bridge_modules import (
    LinearBridgeBaseline,
    ResidualBridge,
    GatedFusionBridge,
    MultiTokenMLP,
    AttentionBridge,
    MiniQFormer,
    QFormer,
    PatchPoolBridge,
    ConvAbstractorBridge,
)

VISION_DIM = 1024
HIDDEN_DIM = 896
NUM_PATCHES = 1024  # InternViT-300M-448px, 1 tile: 32x32 patches (a perfect square)
BATCH = 2


def _assert_bridge_output(bridge: torch.nn.Module, out: torch.Tensor, expected_tokens: int) -> None:
    assert out.shape == (BATCH, expected_tokens, HIDDEN_DIM), \
        f"expected ({BATCH}, {expected_tokens}, {HIDDEN_DIM}), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "bridge output contains NaN/Inf"
    out.sum().backward()
    missing = [n for n, p in bridge.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached these bridge params: {missing}"


# ---- pooled-input bridges: (B, vision_dim) -> (B, num_tokens, hidden_dim) ----

@pytest.mark.parametrize("cls,kwargs,expected_tokens", [
    (LinearBridgeBaseline, {}, None),   # (B, hidden_dim), handled separately below
    (ResidualBridge, {}, None),         # same
    (GatedFusionBridge, {}, None),      # same
    (MultiTokenMLP, {"num_tokens": 8}, 8),
    (MultiTokenMLP, {"num_tokens": 4}, 4),
    (MultiTokenMLP, {"num_tokens": 12}, 12),
])
def test_pooled_bridge_forward(cls, kwargs, expected_tokens):
    bridge = cls(in_features=VISION_DIM, out_features=HIDDEN_DIM, **kwargs)
    x = torch.randn(BATCH, VISION_DIM, requires_grad=False)
    out = bridge(x)
    if expected_tokens is None:
        # single-token bridges return (B, hidden_dim), not (B, 1, hidden_dim)
        assert out.shape == (BATCH, HIDDEN_DIM)
        assert torch.isfinite(out).all()
        out.sum().backward()
        missing = [n for n, p in bridge.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient reached: {missing}"
    else:
        _assert_bridge_output(bridge, out, expected_tokens)


# ---- patch-based bridges: (B, num_patches, vision_dim) -> (B, num_tokens, hidden_dim) ----

@pytest.mark.parametrize("cls,kwargs,expected_tokens", [
    (AttentionBridge, {"num_tokens": 8}, 8),
    (MiniQFormer, {"num_tokens": 8}, 8),  # MiniQFormer reserves 1 for baseline -> output is num_tokens total
    (PatchPoolBridge, {"num_tokens": 8, "pool_type": "mean"}, 8),
    (PatchPoolBridge, {"num_tokens": 8, "pool_type": "max"}, 8),
    (PatchPoolBridge, {"num_tokens": 6, "pool_type": "mean"}, 6),  # non-divisor of 1024, exercises adaptive pool
])
def test_patch_bridge_forward(cls, kwargs, expected_tokens):
    bridge = cls(vision_dim=VISION_DIM, hidden_dim=HIDDEN_DIM, **kwargs)
    x = torch.randn(BATCH, NUM_PATCHES, VISION_DIM)
    out = bridge(x)
    _assert_bridge_output(bridge, out, expected_tokens)


def test_qformer_forward():
    bridge = QFormer(vision_dim=VISION_DIM, hidden_dim=HIDDEN_DIM, num_queries=8, num_layers=1)
    vision = torch.randn(BATCH, NUM_PATCHES, VISION_DIM)
    question = torch.randn(BATCH, 12, HIDDEN_DIM)
    out = bridge(vision, question)
    _assert_bridge_output(bridge, out, 8)


# ---- ConvAbstractorBridge: perfect-square guards + real forward ----

@pytest.mark.parametrize("num_tokens", [4, 9])
def test_conv_abstractor_forward(num_tokens):
    bridge = ConvAbstractorBridge(vision_dim=VISION_DIM, hidden_dim=HIDDEN_DIM,
                                   num_tokens=num_tokens, num_resblocks=1, internal_dim=64)
    x = torch.randn(BATCH, NUM_PATCHES, VISION_DIM)  # 1024 = 32x32, a perfect square
    out = bridge(x)
    _assert_bridge_output(bridge, out, num_tokens)


def test_conv_abstractor_rejects_non_square_num_tokens():
    with pytest.raises(ValueError, match="perfect square"):
        ConvAbstractorBridge(num_tokens=8)  # 8 is not a perfect square


def test_conv_abstractor_rejects_non_square_patch_grid():
    """Multi-tile foot-gun: T*1024 patches (T>1) is only square for square T --
    this must fail loudly at forward(), not silently reshape into garbage."""
    bridge = ConvAbstractorBridge(num_tokens=9, num_resblocks=1, internal_dim=64)
    two_tiles = torch.randn(BATCH, 2 * NUM_PATCHES, VISION_DIM)  # 2048 patches, not a perfect square
    with pytest.raises(ValueError, match="square patch grid"):
        bridge(two_tiles)


# ---- PatchPoolBridge: mean vs max must actually differ ----

def test_patch_pool_mean_and_max_differ():
    torch.manual_seed(0)
    x = torch.randn(BATCH, NUM_PATCHES, VISION_DIM)
    torch.manual_seed(1)
    mean_bridge = PatchPoolBridge(num_tokens=8, pool_type="mean")
    torch.manual_seed(1)
    max_bridge = PatchPoolBridge(num_tokens=8, pool_type="max")
    # same init (same seed) + same input -> outputs must differ (not an accidental no-op)
    out_mean = mean_bridge(x)
    out_max = max_bridge(x)
    assert not torch.allclose(out_mean, out_max), "mean-pool and max-pool produced identical output"


def test_patch_pool_bridge_rejects_bad_pool_type():
    with pytest.raises(ValueError, match="pool_type"):
        PatchPoolBridge(pool_type="sum")
