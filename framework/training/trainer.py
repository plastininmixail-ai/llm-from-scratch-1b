"""Свой Training Loop — без HuggingFace Trainer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from framework.model import GPT, ModelConfig, load_config


@dataclass
class TrainerConfig:
    data_path: str = "data/binary/v0"
    output_dir: str = "checkpoints/v0_1b"
    config_path: str = "configs/v0_1b.yaml"
    max_steps: int = 10000
    batch_size: int = 1
    seq_len: int = 1024
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_accum: int = 16
    log_interval: int = 20
    eval_interval: int = 500
    save_interval: int = 500
    device: str = "cpu"
    seed: int = 42
    resume_from: str = ""  # path to checkpoint for warm-start


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        self.model_cfg = load_config(cfg.config_path)
        self.model = GPT(self.model_cfg).to(cfg.device)
        self.model = torch.compile(self.model) if hasattr(torch, "compile") and cfg.device != "cpu" else self.model

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        # Warm-start: загрузка весов из чекпойнта
        if cfg.resume_from:
            ckpt_path = Path(cfg.resume_from)
            if ckpt_path.exists():
                print(f"[warm-start] Loading {ckpt_path}...")
                ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
                # Удаляем rope.inv_freq (динамический параметр)
                sd = {k: v for k, v in sd.items() if not k.endswith(".inv_freq")}
                missing, unexpected = self.model.load_state_dict(sd, strict=False)
                print(f"[warm-start] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if "step" in ckpt:
                    self.step = ckpt["step"]
                    print(f"[warm-start] Resuming from step {self.step}")
            else:
                print(f"[warm-start] WARNING: {ckpt_path} not found, starting from scratch")

        # Load data
        from framework.data import BinaryDataset
        self.dataset = BinaryDataset(cfg.data_path, cfg.seq_len)
        self.dataloader = self.dataset.get_dataloader(cfg.batch_size)

        # State
        self.step = 0
        self.best_val = float("inf")
        self.log = []

    def get_lr(self, step: int) -> float:
        """Cosine decay с warmup."""
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / self.cfg.warmup_steps
        decay_steps = self.cfg.max_steps - self.cfg.warmup_steps
        if decay_steps <= 0:
            return self.cfg.lr
        progress = (step - self.cfg.warmup_steps) / decay_steps
        import math
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr = self.cfg.lr * self.cfg.min_lr_ratio
        return min_lr + (self.cfg.lr - min_lr) * cosine

    def save_checkpoint(self, is_best: bool = False):
        path = Path(self.cfg.output_dir) / ("best.pt" if is_best else f"step{self.step}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_cfg.__dict__,
            "step": self.step,
            "best_val": self.best_val,
        }, path)
        return path

    def log_step(self, loss: float, elapsed: float):
        entry = {
            "step": self.step,
            "loss": loss,
            "lr": self.get_lr(self.step),
            "elapsed": elapsed,
        }
        self.log.append(entry)
        log_path = Path(self.cfg.output_dir) / "train_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if self.step % self.cfg.log_interval == 0:
            print(f"step {self.step:5d} | loss {loss:.4f} | lr {entry['lr']:.2e} | {elapsed:.0f}s")

    def evaluate(self, n_batches: int = 50) -> float:
        """Оценка на валидационных батчах."""
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.dataloader):
                if i >= n_batches:
                    break
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                _, loss = self.model(x, y)
                total_loss += loss.item()
                count += 1
        self.model.train()
        return total_loss / max(count, 1)

    def train(self) -> dict:
        """Главный training loop."""
        print(f"Training started: {self.cfg.max_steps} steps")
        print(f"Model: {sum(p.numel() for p in self.model.parameters())/1e9:.3f}B params")

        torch.manual_seed(self.cfg.seed)
        self.model.train()

        t0 = time.time()
        data_iter = iter(self.dataloader)

        for step in range(self.cfg.max_steps):
            self.step = step
            # LR
            lr = self.get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            self.optimizer.zero_grad()
            accum_loss = 0
            for micro_step in range(self.cfg.grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    x, y = next(data_iter)
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                _, loss = self.model(x, y)
                loss = loss / self.cfg.grad_accum
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            elapsed = time.time() - t0
            self.log_step(accum_loss, elapsed)

            if step > 0 and step % self.cfg.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"  eval step {step} | val_loss {val_loss:.4f}")
                # Логируем val_loss в train_log
                elapsed = time.time() - self.start_time
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": step,
                        "val_loss": val_loss,
                        "lr": self.get_lr(step),
                        "elapsed": elapsed,
                    }) + "\n")
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint(is_best=True)

            if step > 0 and step % self.cfg.save_interval == 0:
                self.save_checkpoint(is_best=False)

        # Final save
        self.save_checkpoint(is_best=False)
        print(f"Done. best_val={self.best_val:.4f}")
        return {"best_val": self.best_val, "final_step": self.step}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    args = p.parse_args()

    cfg = TrainerConfig(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    trainer = Trainer(cfg)
    result = trainer.train()
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
