"""Тест обучения: 100 шагов с lr=1e-4, без warmup.

Цель: проверить что 1B модель может учиться (loss падает).
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

from framework.training import Trainer, TrainerConfig
from framework.model import load_config

# Используем маленькую модель для теста
cfg = TrainerConfig(
    config_path="configs/v0_1b.yaml",
    data_path="data/binary/v0",
    output_dir="checkpoints/v0_test",
    max_steps=100,
    lr=1e-4,           # меньше для теста
    warmup_steps=10,    # короткий warmup
    batch_size=1,
    seq_len=512,        # короче для скорости
    grad_accum=4,       # меньше для частых обновлений
    save_interval=50,
    eval_interval=50,
    log_interval=10,
)
trainer = Trainer(cfg)
result = trainer.train()
print(f"\nFinal: best_val={result['best_val']:.4f}")
