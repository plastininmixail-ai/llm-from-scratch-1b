"""Training v1 FineWeb-Edu с warm-start от v0."""
import sys
from pathlib import Path
sys.path.insert(0, ".")

from framework.training import Trainer, TrainerConfig

cfg = TrainerConfig(
    config_path="configs/v1_fineweb.yaml",
    data_path="data/binary/v1_fineweb",
    output_dir="checkpoints/v1_fineweb",
    max_steps=5000,
    lr=1.5e-4,
    warmup_steps=200,
    batch_size=1,
    seq_len=1024,
    grad_accum=8,
    save_interval=500,
    eval_interval=200,
    log_interval=20,
    resume_from="checkpoints/v0_1b_1000/best.pt",
)
trainer = Trainer(cfg)
result = trainer.train()
print(f"\nFinal: best_val={result['best_val']:.4f}")
