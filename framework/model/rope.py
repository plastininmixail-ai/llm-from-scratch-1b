"""RoPE — Rotary Position Embedding."""
import math
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cached_cos = emb.cos()[None, None, :, :]
            self._cached_sin = emb.sin()[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        seq_len = q.size(-2)
        self._build_cache(seq_len, q.device)
        cos = self._cached_cos[..., :seq_len, :].to(q.dtype)
        sin = self._cached_sin[..., :seq_len, :].to(q.dtype)

        def rotate(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rotated_x1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
            rotated_x2 = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
            return rotated.flatten(-2)

        return rotate(q), rotate(k)
