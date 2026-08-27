"""
FeedForward (MLP) — два линейных слоя с GELU-активацией.

Архитектура (GPT-2 style):
  FFN(x) = W2 @ dropout(GELU(W1 @ x + b1)) + b2
  hidden = d_model * 4
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4,
                 dropout: float = 0.1, bias: bool = False):
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)  # exact GELU (используется в GPT-2), не tanh-approx
        x = self.fc2(x)
        x = self.dropout(x)
        return x