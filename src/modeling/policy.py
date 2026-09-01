"""Policy network: (P(r|Q), f(I,Q), lambda) -> action  (final-plan P4).

STATUS: skeleton. Imports cleanly; ``__init__`` raises until P4.
"""
from __future__ import annotations

import torch.nn as nn


class PolicyMLP(nn.Module):
    """Small MLP over the discrete action set ``(n_tiles, bridge)``.

    Trained with cross-entropy against the oracle labels ``a*(x, lambda)`` from
    ``outputs/oracle/labels.parquet`` (final-plan D2 / P4).
    """

    def __init__(self, prq_dim: int = 8, visual_dim: int = 64,
                 num_actions: int = 9, hidden: int = 256):
        super().__init__()
        raise NotImplementedError(
            "P4: MLP(prq_dim + visual_dim + 1 -> hidden -> num_actions); "
            "input is (P(r|Q), f(I,Q), lambda)."
        )
