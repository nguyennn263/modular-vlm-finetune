import torch
import torch.nn as nn

from src.modeling.bridge_modules import Mlp1Bridge, pixel_shuffle_v2
from src.training.setup import VisionLanguageBridge


def _mlp1():
    # Vintern-1B-v3_5's mlp1 (modeling_internvl_chat.py): LN(1024*4) -> Linear -> GELU -> Linear
    return nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 896), nn.GELU(), nn.Linear(896, 896))


def _vit_out(b=2):
    torch.manual_seed(0)
    return torch.randn(b, 1 + 24 * 24, 1024)    # 336px: CLS + 24x24 patches


def test_pixel_shuffle_groups_2x2_neighbours():
    x = torch.arange(4 * 4, dtype=torch.float32).view(1, 4, 4, 1)
    y = pixel_shuffle_v2(x)                     # (1, 2, 2, 4)
    assert y.shape == (1, 2, 2, 4)
    assert sorted(y[0, 0, 0].tolist()) == [0.0, 1.0, 4.0, 5.0]   # top-left 2x2 block


def test_mlp1_variant_is_vintern_projector_on_shuffled_patches():
    m = _mlp1()
    out = Mlp1Bridge(m)(_vit_out())
    x = _vit_out()[:, 1:].reshape(2, 24, 24, 1024)
    assert out.shape == (2, 144, 896)
    assert torch.allclose(out, m(pixel_shuffle_v2(x).reshape(2, -1, 4096)))


def test_residual_starts_exactly_at_mlp1():
    m = _mlp1()
    assert torch.allclose(Mlp1Bridge(m, residual_dim=256)(_vit_out()), Mlp1Bridge(m)(_vit_out()))


def test_hybrid_prepends_global_tokens_and_trains_only_new_parts():
    b = Mlp1Bridge(_mlp1(), residual_dim=256, num_global_tokens=8)
    out = b(_vit_out())
    assert out.shape == (2, 8 + 144, 896)
    assert torch.allclose(out[:, :8], b.global_tokens(_vit_out()[:, 0]))
    assert not any(p.requires_grad for p in b.mlp1.parameters())
    trainable = sum(p.numel() for p in b.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in b.delta.parameters()) + \
        sum(p.numel() for p in b.global_tokens.parameters())


class _FakeVintern(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(2, 2)
        self.language_model = nn.Linear(2, 2)
        self.mlp1 = _mlp1()


def test_other_bridges_do_not_pick_up_mlp1_parameters():
    w = VisionLanguageBridge(_FakeVintern(), "multi_token")
    names = [n for n, _ in w.named_parameters()]
    assert not any("mlp1" in n for n in names)
    trainable = sum(p.numel() for p in w.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in w.bridge.parameters())


def test_hybrid_wrapper_keeps_mlp1_frozen():
    w = VisionLanguageBridge(_FakeVintern(), "hybrid", {"residual_dim": 256, "num_global_tokens": 8})
    frozen = [n for n, p in w.named_parameters() if "mlp1" in n]
    assert frozen and not any(p.requires_grad for n, p in w.named_parameters() if "mlp1" in n)
