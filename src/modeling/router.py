"""Router: cognitive prior P(r|Q) + cheap visual state f(I,Q)  (final-plan P4, contribution #2).

P(r|Q): predict the reasoning-type distribution from the QUESTION alone — a PhoBERT
encoder + a linear head over the 8 canonical categories. Trained with a
class-balanced loss on the grouped-split `category` labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.reasoning_types import CAT2IDX, CATEGORIES, IDX2CAT  # noqa: F401 (re-export)


class PrQHead(nn.Module):
    """P(r|Q) — reasoning-type logits from the question text."""

    def __init__(self, encoder_name: str = "vinai/phobert-base-v2",
                 num_classes: int = len(CATEGORIES), dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # <s>
        return self.head(self.dropout(cls))  # (B, num_classes)

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask) -> torch.Tensor:
        return torch.softmax(self(input_ids, attention_mask), dim=-1)


class VisualStateProbe(nn.Module):
    """f(I,Q): cheap visual-state features that do NOT use the category label.

    Plan: pooled InternViT features at n_tiles=1 + a few cheap metadata signals
    (image clarity/occlusion/object-density, question length). Kept small so it is
    negligible next to the main pipeline.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU())

    def forward(self, pooled_vision: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_vision)
