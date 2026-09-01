"""Policy network: (P(r|Q), f(I,Q), λ) -> action  (final-plan P4)."""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    """Small MLP over the discrete action set ``(n_tiles, bridge)``.

    Trained with cross-entropy against the oracle labels a*(x, λ).
    ``f(I,Q)`` is optional: pass visual_dim=0 for the Reasoning-type-only ablation.
    """

    def __init__(self, prq_dim: int = 8, visual_dim: int = 64,
                 num_actions: int = 9, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        assert prq_dim or visual_dim, "policy needs at least one of P(r|Q) / f(I,Q)"
        self.prq_dim, self.visual_dim = prq_dim, visual_dim
        in_dim = prq_dim + visual_dim + 1  # +1 for lambda
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, prq: torch.Tensor | None, lam: torch.Tensor,
                fiq: torch.Tensor | None = None) -> torch.Tensor:
        ref = prq if prq is not None else fiq
        lam = lam.reshape(ref.shape[0], 1).to(ref.dtype)
        parts = []
        if self.prq_dim:
            assert prq is not None, "prq_dim>0 but P(r|Q) not given"
            parts.append(prq)
        if self.visual_dim:
            assert fiq is not None, "visual_dim>0 but f(I,Q) not given"
            parts.append(fiq)
        parts.append(lam)
        return self.net(torch.cat(parts, dim=-1))
