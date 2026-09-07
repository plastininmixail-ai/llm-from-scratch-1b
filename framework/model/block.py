"""DecoderBlock — Pre-LN transformer block."""
import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import FeedForward
from .rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 mlp_mult: int = 4, dropout: float = 0.0, bias: bool = False, rope=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout, bias, rope=rope)
        self.ln2 = RMSNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_mult, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
