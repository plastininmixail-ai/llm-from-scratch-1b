"""GPT — decoder-only transformer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .block import DecoderBlock
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding


class GPT(nn.Module):
    """GPT-style decoder-only transformer с RMSNorm + RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        if config.rope_enabled:
            self.rope = RotaryEmbedding(
                dim=config.d_model // config.n_heads,
                max_seq_len=config.max_seq_len,
                base=config.rope_base,
            )
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.rope = None

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                mlp_mult=config.mlp_hidden_mult,
                dropout=config.dropout,
                bias=config.bias,
                rope=self.rope,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.zeros_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"context {T} > max {self.config.max_seq_len}"

        tok = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = self.dropout(tok + self.pos_emb(pos))
        else:
            x = self.dropout(tok)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        """Простая генерация (для inference agent)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                mask = cum - sorted_logits.softmax(-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
