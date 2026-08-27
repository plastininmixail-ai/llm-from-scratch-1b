"""
Multi-Head Causal Self-Attention с предвычисленной causal mask.

Архитектура:
  Q, K, V = Linear(x) → split на h голов → (B, h, T, head_dim)
  attn = softmax(QK^T / sqrt(head_dim) + mask) @ V
  out = Linear(concat heads)

Это GPT-2-style MHA — все головы в одном Linear (QKV вместе).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # fused QKV projection (GPT-2 style: 3 матрицы одним matmul)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask: -inf выше главной диагонали
        # регистрируем как buffer (не параметр, не сохраняется отдельно)
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → out: (B, T, d_model)"""
        B, T, C = x.shape

        # (B, T, 3*d_model) → 3 × (B, T, d_model)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv  # каждый (B, T, d_model)

        # split на головы: (B, T, d_model) → (B, T, n_heads, head_dim) → (B, h, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention: (B, h, T, head_dim) @ (B, h, head_dim, T)
        # → (B, h, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # causal mask: верхний треугольник → -inf
        attn = attn.masked_fill(self.mask[:T, :T] == float("-inf"), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # (B, h, T, T) @ (B, h, T, head_dim) → (B, h, T, head_dim)
        y = attn @ v

        # concat голов: (B, h, T, head_dim) → (B, T, h, head_dim) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.proj(y))
        return y