"""
Training loop для GPT на CPU.

Особенности:
  - AdamW optimizer (PyTorch встроенный)
  - LR schedule: warmup + cosine decay
  - Gradient clipping (max_norm=1.0) — стабилизация
  - Периодические eval (loss на валидационном сплите)
  - Сохранение/загрузка чекпоинтов (model + optimizer + step)
  - Логирование в JSON-лог
  - Простой throughput (tokens/sec)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import torch

# CPU-оптимизации: ограничиваем потоки (8 cores на этом CPU)
try:
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)
    print(f"[cpu] torch.set_num_threads={n_threads}")
except Exception:
    pass
from torch.utils.data import DataLoader

from model import GPT


@dataclass
class TrainerConfig:
    # Оптимизация
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 200
    max_steps: int = 5000
    min_lr_ratio: float = 0.1  # минимальный lr = min_lr_ratio * lr

    # Batching
    batch_size: int = 16
    grad_accum_steps: int = 1  # для effective batch = batch_size * grad_accum_steps

    # Eval / log
    eval_interval: int = 100
    log_interval: int = 20
    eval_iters: int = 20

    # Saving
    save_interval: int = 1000
    out_dir: str = "checkpoints"

    # Device
    device: str = "cpu"

    # Seed
    seed: int = 42


def get_lr(step: int, cfg: TrainerConfig) -> float:
    """Cosine schedule с linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


class Trainer:
    def __init__(self, model: GPT, train_ds, val_ds, cfg: TrainerConfig,
                 resume_from: str | Path | None = None,
                 reset_step: bool = False,
                 resume_optimizer: bool = False):
        self.model = model.to(cfg.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.step = 0
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        torch.manual_seed(cfg.seed)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "train_log.jsonl"
        # создаём файл сразу
        self.log_path.write_text("")
        # метрики
        self.best_val = float("inf")

        # warm-start: загрузить model+optimizer+step из чекпойнта
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=cfg.device,
                              weights_only=False)

            # === verify_checkpoint: проверка целостности ===
            required = ["model"]
            missing = [k for k in required if k not in ckpt]
            if missing:
                raise RuntimeError(
                    f"[verify_checkpoint] FAIL: missing keys: {missing}")
            # проверка embedding shape (vocab × d_model)
            for k in ("tok_emb.weight", "tok_embeddings.weight"):
                if k in ckpt["model"]:
                    emb_shape = tuple(ckpt["model"][k].shape)
                    cfg_vocab = self.model.config.vocab_size
                    cfg_dim = self.model.config.d_model
                    if emb_shape != (cfg_vocab, cfg_dim):
                        raise RuntimeError(
                            f"[verify_checkpoint] FAIL: embedding shape "
                            f"{emb_shape} != ({cfg_vocab}, {cfg_dim})")
                    break
            print(f"[verify_checkpoint] ✓ {resume_from} "
                  f"(step={ckpt.get('step', '?')}, "
                  f"best_val={ckpt.get('best_val', float('nan')):.4f})")

            # === validate_architecture: сравнение с текущей конфигурацией ===
            arch = ckpt.get("model_config", {})
            checks = {
                "d_model": arch.get("d_model"),
                "n_heads": arch.get("n_heads"),
                "n_layers": arch.get("n_layers"),
                "vocab_size": arch.get("vocab_size"),
                "max_seq_len": arch.get("max_seq_len"),
            }
            mismatches = []
            cur = self.model.config
            for k, saved in checks.items():
                cur_v = getattr(cur, k, None)
                if saved is not None and saved != cur_v:
                    mismatches.append(f"{k}: ckpt={saved} != current={cur_v}")
            if mismatches:
                raise RuntimeError(
                    f"[validate_architecture] FAIL: {mismatches}")
            print(f"[validate_architecture] ✓ config matches")

            self.model.load_state_dict(ckpt["model"])
            self.step = ckpt.get("step", 0)
            if reset_step:
                old_step = self.step
                self.step = 0
                print(f"[warm-start] reset-step: {old_step} → 0 (веса сохранены)")
            if resume_optimizer and "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    print("[warm-start] optimizer state загружен")
                except Exception as e:
                    print(f"[warn] не удалось загрузить optimizer: {e}")
            # восстановить best_val если есть (для логики "best")
            self.best_val = ckpt.get("best_val", float("inf"))
            print(f"[warm-start] загружен чекпойнт {resume_from}, "
                  f"step={self.step}, best_val={self.best_val:.4f}")

    def _make_loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,  # CPU
            pin_memory=False,
            drop_last=True,
        )

    def _eval(self) -> float:
        self.model.eval()
        loader = self._make_loader(self.val_ds, shuffle=False)
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= self.cfg.eval_iters:
                    break
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    def _save(self, tag: str = "ckpt") -> None:
        path = self.out_dir / f"{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "step": self.step,
            "model_config": asdict(self.model.config),
            "trainer_config": asdict(self.cfg),
        }, path)

    def _log(self, **kwargs) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")

    def train(self) -> None:
        """Главный training loop."""
        cfg = self.cfg
        print(f"[train] device={cfg.device}, batch_size={cfg.batch_size}, "
              f"max_steps={cfg.max_steps}, lr={cfg.lr}")
        print(f"[train] train_examples={len(self.train_ds)}, "
              f"val_examples={len(self.val_ds)}")
        print(f"[train] params={self.model.num_parameters()/1e6:.2f}M")
        print(f"[train] log → {self.log_path}")

        train_loader = self._make_loader(self.train_ds, shuffle=True)
        train_iter = iter(train_loader)

        self.model.train()
        t0 = time.time()
        loss_accum = 0.0

        while self.step < cfg.max_steps:
            # set lr
            lr = get_lr(self.step, cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # forward + backward
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(cfg.device), y.to(cfg.device)
            _, loss = self.model(x, y)
            loss.backward()
            loss_accum += loss.item()

            # gradient accumulation
            if (self.step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.step += 1

            # log
            if self.step % cfg.log_interval == 0:
                avg_loss = loss_accum / cfg.log_interval
                loss_accum = 0.0
                elapsed = time.time() - t0
                tok_per_sec = (cfg.batch_size * cfg.log_interval * T_estimate(cfg)) / elapsed if False else 0
                tokens_done = self.step * cfg.batch_size * T_estimate(cfg)
                tok_per_sec = tokens_done / elapsed if elapsed > 0 else 0
                msg = (f"step {self.step}/{cfg.max_steps} | "
                       f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                       f"{tok_per_sec:.0f} tok/s | {elapsed:.0f}s")
                print(msg, flush=True)
                self._log(
                    type="train", step=self.step, loss=avg_loss,
                    lr=lr, tok_per_sec=tok_per_sec, elapsed=elapsed,
                )
                t0 = time.time()  # reset for next window

            # eval
            if self.step % cfg.eval_interval == 0:
                val_loss = self._eval()
                print(f"  eval step {self.step} | val_loss {val_loss:.4f}", flush=True)
                self._log(type="eval", step=self.step, val_loss=val_loss)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save("best")

            # save
            if self.step % cfg.save_interval == 0:
                self._save(f"step{self.step}")
                print(f"  saved step{self.step}.pt", flush=True)

        # финальное сохранение
        self._save("final")
        print(f"[train] done. best_val={self.best_val:.4f}")


def T_estimate(cfg: TrainerConfig) -> int:
    """Примерная длина seq для подсчёта throughput. Реально берётся из dataset."""
    return 512  # default, заменится на настоящее значение при вызове
