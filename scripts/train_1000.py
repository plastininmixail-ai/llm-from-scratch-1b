"""Полный pretrain 1B модели: 1000 шагов."""
import sys
from pathlib import Path
sys.path.insert(0, ".")

from framework.training import Trainer, TrainerConfig

cfg = TrainerConfig(
    config_path="configs/v0_1b.yaml",
    data_path="data/binary/v0",
    output_dir="checkpoints/v0_1b_1000",
    max_steps=1000,
    lr=3e-4,
    warmup_steps=100,
    batch_size=1,
    seq_len=1024,
    grad_accum=8,
    save_interval=200,
    eval_interval=100,
    log_interval=20,
)
trainer = Trainer(cfg)
result = trainer.train()
print(f"\nFinal: best_val={result['best_val']:.4f}")
