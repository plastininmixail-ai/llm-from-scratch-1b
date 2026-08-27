"""
GPT-style decoder-only трансформер:
  - Token embeddings + learned positional embeddings
  - N x DecoderBlock (Pre-LN)
  - Final LayerNorm
  - LM Head (Linear → vocab_size logits)
  - Weight tying: output head shares weights with token embeddings
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import DecoderBlock


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                mlp_mult=config.mlp_hidden_mult,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying: head и tok_emb делят матрицу
        # (это уменьшает параметры и помогает обучению)
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        # инициализация
        self.apply(self._init_weights)

        # специальная инициализация для residual-проекций (как в GPT-2)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                # scaled init для residual (улучшает стабильность)
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx: (B, T) — токены
        targets: (B, T) — те же токены, сдвинутые на 1 (опционально)
        returns: (logits, loss)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"контекст {T} > max_seq_len {self.config.max_seq_len}"
        )

        # позиции [0..T)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.tok_emb(idx)            # (B, T, d_model)
        pos_emb = self.pos_emb(pos)         # (T, d_model)
        x = self.dropout(tok + pos_emb)    # broadcast (T, d_model) → (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # стандартный LM loss: предсказываем targets[t] по logits[t]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Простая greedy/sampling генерация.
        idx: (B, T) — стартовые токены
        returns: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # обрезаем контекст до max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len \
                else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            # берём логиты последнего токена
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    # ------------------------------------------------------------------ #
    # Утилиты
    # ------------------------------------------------------------------ #
    def num_parameters(self, exclude_tied: bool = True) -> int:
        """Общее число параметров; exclude_tied=True — без output head (он shared)."""
        n = sum(p.numel() for p in self.parameters())
        if exclude_tied:
            n -= self.head.weight.numel()
        return n

    def info(self) -> str:
        n = self.num_parameters()
        return (f"GPT(vocab={self.config.vocab_size}, d_model={self.config.d_model}, "
                f"n_heads={self.config.n_heads}, n_layers={self.config.n_layers}, "
                f"max_seq_len={self.config.max_seq_len}) — {n / 1e6:.1f}M параметров")