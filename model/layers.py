"""
DecoderBlock — Pre-LN версия (как в современных LLM):

  x = x + attn(layer_norm(x))
  x = x + mlp(layer_norm(x))

Pre-LN стабильнее для глубоких сетей (без прогрева lr).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 mlp_mult: int = 4, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout, bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_mult, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x