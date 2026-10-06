import pytest
import torch
import torch.nn as nn

from src.data.collator import IMG_CONTEXT_TOKEN, image_placeholder
from src.modeling.bridge_modules import (GlobalLocalBridge, MultiTokenMLP, fill_image_slots,
                                         pixel_shuffle_v2)
from src.training.setup import VisionLanguageBridge


def _mlp1():
    # Vintern-1B-v3_5's mlp1: LN(1024*4) -> Linear -> GELU -> Linear
    return nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 896), nn.GELU(), nn.Linear(896, 896))


def _vit_out(b=2):
    torch.manual_seed(0)
    return torch.randn(b, 1 + 24 * 24, 1024)    # 336px: CLS + 24x24 patches


def _all_mlp1_tokens(m, x):
    b = x.shape[0]
    return m(pixel_shuffle_v2(x[:, 1:].reshape(b, 24, 24, 1024)).reshape(b, 144, 4096))


def test_full_grid_is_vintern_projector_and_global_is_multitoken_on_cls():
    m, x = _mlp1(), _vit_out()
    br = GlobalLocalBridge(m, num_tokens=14, local_grid=12)
    glob, local = br(x)
    assert glob.shape == (2, 14, 896) and local.shape == (2, 144, 896)
    assert torch.allclose(local, _all_mlp1_tokens(m, x))
    assert torch.allclose(glob, br.global_tokens(x[:, 0]))


def test_pooled_grid_averages_neighbouring_tokens_row_major():
    m, x = _mlp1(), _vit_out()
    full = _all_mlp1_tokens(m, x)                              # token t <-> (t // 12, t % 12)
    _, local6 = GlobalLocalBridge(m, local_grid=6)(x)
    assert local6.shape == (2, 36, 896)
    assert torch.allclose(local6[:, 0], full[:, [0, 1, 12, 13]].mean(1), atol=1e-5)
    assert torch.allclose(local6[:, 1], full[:, [2, 3, 14, 15]].mean(1), atol=1e-5)    # next column
    assert torch.allclose(local6[:, 6], full[:, [24, 25, 36, 37]].mean(1), atol=1e-5)  # next row
    _, local3 = GlobalLocalBridge(m, local_grid=3)(x)
    block = [r * 12 + c for r in range(4) for c in range(4)]
    assert local3.shape == (2, 9, 896)
    assert torch.allclose(local3[:, 0], full[:, block].mean(1), atol=1e-5)
    _, local1 = GlobalLocalBridge(m, local_grid=1)(x)
    assert torch.allclose(local1[:, 0], full.mean(1), atol=1e-5)


def test_grid_zero_has_no_local_path():
    glob, local = GlobalLocalBridge(_mlp1(), num_tokens=8, local_grid=0)(_vit_out())
    assert local is None and glob.shape == (2, 8, 896)


def test_only_the_global_tokens_train():
    br = GlobalLocalBridge(_mlp1(), num_tokens=14, local_grid=6)
    trainable = sum(p.numel() for p in br.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in MultiTokenMLP(1024, 896, 14).parameters())
    assert not any(p.requires_grad for p in br.mlp1.parameters())
    _, local = br(_vit_out())
    assert not local.requires_grad


def test_fill_image_slots_handles_left_padding_and_checks_counts():
    ctx = 99
    ids = torch.tensor([[0, 0, 5, 99, 99, 7],      # left-padded row
                        [5, 99, 99, 7, 8, 9]])
    text = torch.zeros(2, 6, 3)
    local = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
    out = fill_image_slots(text, ids, ctx, local)
    assert torch.equal(out[0, 3], local[0, 0]) and torch.equal(out[0, 4], local[0, 1])
    assert torch.equal(out[1, 1], local[1, 0]) and torch.equal(out[1, 2], local[1, 1])
    assert out[0, :3].abs().sum() == 0 and out[1, 3:].abs().sum() == 0   # nothing else touched
    assert text.abs().sum() == 0                                         # input not modified
    with pytest.raises(ValueError):
        fill_image_slots(text, ids, ctx, torch.zeros(2, 3, 3))


def test_image_placeholder():
    assert image_placeholder(0) == "<image>"
    assert image_placeholder(3) == "<img>" + IMG_CONTEXT_TOKEN * 3 + "</img>"


class _FakeVintern(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(2, 2)
        self.language_model = nn.Linear(2, 2)
        self.mlp1 = _mlp1()


@pytest.mark.parametrize("grid,expected", [(12, 144), (6, 36), (3, 9), (1, 1), (0, 0)])
def test_wrapper_reports_slot_size(grid, expected):
    w = VisionLanguageBridge(_FakeVintern(), "global_local", {"num_tokens": 8, "local_grid": grid})
    assert w.n_image_slot_tokens == expected
    assert w.uses_patches


def test_other_bridges_have_no_image_slot():
    assert VisionLanguageBridge(_FakeVintern(), "multi_token").n_image_slot_tokens == 0


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True,
                                             use_fast=False)
    except Exception as exc:  # no cached tokenizer / no network
        pytest.skip(f"Vintern tokenizer unavailable: {exc}")


@pytest.mark.parametrize("k", [0, 9, 144])
def test_collator_slot_and_answer_mask(k, tmp_path):
    from PIL import Image
    from src.data.collator import custom_collate_fn
    from src.schema.data_schema import OneSample

    tok = _tokenizer()
    img = tmp_path / "x.jpg"
    Image.new("RGB", (40, 30)).save(img)
    batch = [OneSample(image_path=str(img), question="Người đàn ông đang làm gì?",
                       answers=["Đang chơi tennis trên sân"]),
             OneSample(image_path=str(img), question="Màu áo?", answers=["Màu đỏ"])]
    out = custom_collate_fn(batch, tokenizer=tok, max_length=256 + k, image_slot_tokens=k)
    ctx = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    assert (out["input_ids"] == ctx).sum(1).tolist() == [k, k]
    for row, ans in enumerate(["Đang chơi tennis trên sân", "Màu đỏ"]):
        start = int(out["answer_start_pos"][row])
        n = int(out["attention_mask"][row].sum())
        decoded = tok.decode(out["input_ids"][row, start:n], skip_special_tokens=True)
        assert decoded.strip() == ans     # loss positions = the answer, nothing truncated
