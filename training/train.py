"""
CLI для обучения GPT на локальном корпусе.

Использование:
    python -m training.train \
        --config configs/small.yaml \
        --tokenizer tokenizer/vocab.json \
        --corpus data/raw/corpus.txt \
        --max-steps 2000 \
        --batch-size 8 \
        --seq-len 256 \
        --out-dir checkpoints/small-v1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import random_split

from model import GPT, load_config
from tokenizer.bpe import BPETokenizer
from training.dataset import TextDataset
from training.trainer import Trainer, TrainerConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    # токенизатор
    tok = BPETokenizer.load(args.tokenizer)
    print(f"tokenizer: {tok.info()}")

    # модель
    cfg = load_config(args.config, vocab_size=len(tok))
    model = GPT(cfg)
    print(f"model: {model.info()}")

    # датасет
    ds = TextDataset(args.corpus, tok, seq_len=args.seq_len, stride=args.seq_len)
    print(f"dataset: {ds.info()}")
    n_val = max(50, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"split: train={n_train:,} val={n_val:,}")

    # trainer config
    tcfg = TrainerConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        out_dir=str(args.out_dir),
        device=args.device,
    )

    trainer = Trainer(model, train_ds, val_ds, tcfg)
    t0 = time.time()
    trainer.train()
    print(f"total time: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())