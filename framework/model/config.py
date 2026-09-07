"""ModelConfig — конфигурация модели."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 11
    max_seq_len: int = 1024
    dropout: float = 0.0
    mlp_hidden_mult: int = 4
    rope_enabled: bool = True
    rope_base: float = 10000.0
    tie_weights: bool = True
    bias: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def load_config(path: str | Path) -> ModelConfig:
    """Загружает YAML конфиг в ModelConfig."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    m = data["model"]
    return ModelConfig(
        vocab_size=m.get("vocab_size", 32000),
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        max_seq_len=m["max_seq_len"],
        dropout=m.get("dropout", 0.0),
        rope_enabled=m.get("rope_enabled", True),
        rope_base=m.get("rope_base", 10000.0),
        tie_weights=m.get("tie_weights", True),
        bias=m.get("bias", False),
    )


def count_params(cfg: ModelConfig) -> int:
    """Подсчёт параметров модели."""
    d = cfg.d_model
    L = cfg.n_layers
    V = cfg.vocab_size

    # Embedding (будет tied с output, считаем один раз)
    emb = V * d if not cfg.tie_weights else 0
    emb_tied = V * d  # для расчёта общего размера

    # Per layer:
    # Attention: QKV (3*d²) + Proj (d²) = 4*d²
    attn = 4 * d * d
    # MLP: fc1 (d*4d) + fc2 (4d*d) = 8*d²
    mlp = 8 * d * d
    layer = attn + mlp

    # Final RMSNorm
    final_norm = d

    total = emb + emb_tied + L * layer + final_norm
    return total
