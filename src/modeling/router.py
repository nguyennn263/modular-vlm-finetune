"""Router: cognitive prior P(r|Q) + cheap visual state f(I,Q)  (final-plan P4).

STATUS: skeleton. The classes import cleanly; ``__init__`` raises until P4 wires them up.
"""
from __future__ import annotations

import torch.nn as nn

CATEGORIES = [
    "relational", "recognition", "spatial", "causal",
    "action", "counting", "context", "yesno",
]


class PrQHead(nn.Module):
    """P(r|Q): reasoning-type distribution predicted from the QUESTION alone.

    Plan: PhoBERT-base-v2 encoder (frozen or lightly tuned) + linear head over
    the 8 canonical categories, class-balanced cross-entropy on ``category``.
    """

    def __init__(self, encoder_name: str = "vinai/phobert-base-v2",
                 num_classes: int = len(CATEGORIES)):
        super().__init__()
        raise NotImplementedError(
            "P4: load PhoBERT, add an 8-way head, train on data/splits/train.jsonl "
            "category labels with inverse-frequency class weights."
        )


class VisualStateProbe(nn.Module):
    """f(I,Q): cheap visual-state features, NOT using the category label.

    Plan: pooled InternViT features at n_tiles=1 concatenated with a few cheap
    metadata signals (image clarity/occlusion/object-density, question length).
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        raise NotImplementedError("P4: define the cheap probe head")
