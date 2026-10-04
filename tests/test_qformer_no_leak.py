import torch

from src.modeling.bridge_modules import QFormer


def _inputs(answer_value: float):
    torch.manual_seed(0)
    vision = torch.randn(2, 10, 1024)
    text = torch.randn(2, 7, 896)
    text[:, 4:] = answer_value          # positions 4..6 play the reference answer
    hidden = torch.zeros(2, 7, dtype=torch.bool)
    hidden[:, 4:] = True                # what trainer.forward_pass masks (answer_start_pos = 4)
    return vision, text, hidden


def test_masked_answer_does_not_reach_the_bridge():
    torch.manual_seed(1)
    q = QFormer().eval()
    a = q(*_inputs(1.0))
    b = q(*_inputs(-3.0))
    assert torch.allclose(a, b, atol=1e-5)


def test_unmasked_answer_does_reach_the_bridge():
    torch.manual_seed(1)
    q = QFormer().eval()
    v1, t1, _ = _inputs(1.0)
    v2, t2, _ = _inputs(-3.0)
    assert not torch.allclose(q(v1, t1), q(v2, t2), atol=1e-5)
