"""
Тесты архитектуры модели.
Запуск: pytest tests/test_model.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from model import GPT, ModelConfig, load_config


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        dropout=0.0,
        tie_weights=True,
    )


@pytest.fixture
def model(cfg):
    return GPT(cfg)


# ---------------------------------------------------------------------------- #
# Базовые тесты
# ---------------------------------------------------------------------------- #
class TestModelBasics:
    def test_model_created(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0

    def test_num_parameters_excludes_tied(self, model, cfg):
        """head.weight должен быть shared с tok_emb.weight."""
        # с учётом weight tying: count = tok_emb + pos_emb + blocks + ln_f
        # БЕЗ head (он shared)
        n = model.num_parameters(exclude_tied=True)
        # проверим: param count должен быть меньше, чем общее число параметров
        total = sum(p.numel() for p in model.parameters())
        assert n < total
        # разница = vocab * d_model
        diff = total - n
        assert diff == cfg.vocab_size * cfg.d_model


class TestForwardPass:
    def test_forward_shape(self, model, cfg):
        B, T = 2, 16
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(idx)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is None  # targets не переданы

    def test_forward_with_targets(self, model, cfg):
        B, T = 2, 16
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(idx, targets)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is not None
        assert loss.item() > 0  # loss всегда положительный

    def test_loss_is_finite(self, model, cfg):
        idx = torch.randint(0, cfg.vocab_size, (4, 8))
        targets = torch.randint(0, cfg.vocab_size, (4, 8))
        _, loss = model(idx, targets)
        assert torch.isfinite(loss).all().item()

    def test_causal_mask_respected(self, model, cfg):
        """Токен на позиции T не должен зависеть от токенов на позициях > T."""
        # возьмём 2 одинаковых контекста, но с разными последними токенами
        # первый контекст: [a, a, a, a, X]
        # второй контекст: [a, a, a, a, Y]
        # логиты для первых 4 позиций должны быть одинаковыми
        T = 8
        idx1 = torch.zeros(1, T, dtype=torch.long)
        idx1[0, -1] = 100
        idx2 = torch.zeros(1, T, dtype=torch.long)
        idx2[0, -1] = 200

        logits1, _ = model(idx1)
        logits2, _ = model(idx2)
        # logits[:, :-1, :] должны быть одинаковыми (только последний токен разный)
        assert torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-5), (
            "causal mask не работает: позиция 0..T-2 зависит от будущего токена T-1"
        )


class TestGeneration:
    def test_generate_shape(self, model, cfg):
        B, T = 1, 5
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        out = model.generate(idx, max_new_tokens=10)
        assert out.shape == (B, T + 10)

    def test_generate_with_top_k(self, model, cfg):
        idx = torch.randint(0, cfg.vocab_size, (1, 5))
        out = model.generate(idx, max_new_tokens=5, top_k=10, temperature=0.5)
        assert out.shape == (1, 10)


class TestConfig:
    def test_load_default(self):
        cfg = ModelConfig()
        assert cfg.d_model % cfg.n_heads == 0

    def test_load_from_yaml(self):
        cfg = load_config("configs/small.yaml", vocab_size=8000)
        assert cfg.vocab_size == 8000
        assert cfg.d_model % cfg.n_heads == 0

    def test_head_dim_property(self):
        cfg = ModelConfig(d_model=384, n_heads=6)
        assert cfg.head_dim == 64


class TestModelSize:
    """Проверяем, что размер модели соответствует ожиданиям."""

    def test_small_model_size(self):
        # Размер берётся из configs/small.yaml (после обновления)
        # и валидируется против реального числа параметров.
        from model import load_config
        cfg = load_config("configs/small.yaml", vocab_size=1500)
        # ожидаемое значение: ~3.3M для d_model=256, n_layers=4
        # допуск: ±20% от теоретической формулы (с учётом bias=False в GPT-2)
        # формула: tok_emb + pos_emb + n_layers*(4*d² + 2*d*hidden) + 2*d
        expected = (
            cfg.vocab_size * cfg.d_model
            + cfg.max_seq_len * cfg.d_model
            + cfg.n_layers * (
                4 * cfg.d_model ** 2
                + 2 * cfg.d_model * cfg.d_model * cfg.mlp_hidden_mult
            )
            + 2 * cfg.d_model  # ln_f
        )
        n = GPT(cfg).num_parameters(exclude_tied=True)
        # даём допуск ±15% из-за bias/batch-stats, не учтённых в формуле
        assert abs(n - expected) / expected < 0.15, (
            f"ожидалось ~{expected/1e6:.2f}M, получили {n/1e6:.2f}M "
            f"(разница {abs(n-expected)/expected*100:.1f}%)"
        )