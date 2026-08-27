"""
Загрузчик конфигурации модели из YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    mlp_hidden_mult: int = 4
    rope_enabled: bool = False
    rope_base: float = 10000.0
    tie_weights: bool = True
    bias: bool = False  # GPT-2 style: нет bias в Linear

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) должно делиться на n_heads ({self.n_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def load_config(path: str | Path, vocab_size: Optional[int] = None) -> ModelConfig:
    """Загружает YAML и создаёт ModelConfig."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    m = data["model"]
    mlp = data.get("mlp", {})
    rope = data.get("rope", {})

    cfg = ModelConfig(
        vocab_size=vocab_size if vocab_size is not None else 8000,
        d_model=m.get("d_model", 384),
        n_heads=m.get("n_heads", 6),
        n_layers=m.get("n_layers", 6),
        max_seq_len=m.get("max_seq_len", 512),
        dropout=m.get("dropout", 0.1),
        mlp_hidden_mult=mlp.get("hidden_dim_mult", 4),
        rope_enabled=rope.get("enabled", False),
        rope_base=rope.get("base", 10000.0),
    )
    return cfg