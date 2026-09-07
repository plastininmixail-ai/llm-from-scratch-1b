"""Свой Evaluation Framework — без HuggingFace."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np


from ..model import GPT, load_config


@torch.inference_mode()
def compute_perplexity(
    model: GPT,
    tokens: torch.Tensor,
    seq_len: int = 512,
    device: str = "cpu",
) -> float:
    """Считает perplexity на массиве токенов.

    Args:
        model: GPT
        tokens: (N,) тензор токенов
        seq_len: длина окна для оценки
        device: 'cpu' / 'cuda'

    Returns:
        perplexity (float)
    """
    model.eval()
    tokens = tokens.to(device)
    n = tokens.shape[0]
    n_chunks = n // seq_len

    if n_chunks < 1:
        raise ValueError(f"Need at least {seq_len} tokens, got {n}")

    losses = []
    for i in range(n_chunks):
        chunk = tokens[i * seq_len:(i + 1) * seq_len]
        x = chunk[:-1].unsqueeze(0)  # (1, T-1)
        y = chunk[1:].unsqueeze(0)   # (1, T-1)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="mean",
        )
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    perplexity = float(np.exp(mean_loss))
    return perplexity


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    eval_data_path: Path,
    seq_len: int = 512,
    max_chunks: int = 100,
    device: str = "cpu",
) -> dict:
    """Полная оценка: perplexity + базовые метрики."""
    from ..inference import load_model_for_inference

    model = load_model_for_inference(checkpoint_path, config_path, device=device)

    # Загружаем eval данные (uint16 binary)
    tokens = np.fromfile(eval_data_path, dtype=np.uint16).astype(np.int64)
    tokens = torch.from_numpy(tokens)
    n_chunks = min(max_chunks, len(tokens) // seq_len)
    tokens = tokens[:n_chunks * seq_len]

    ppl = compute_perplexity(model, tokens, seq_len=seq_len, device=device)

    return {
        "checkpoint": str(checkpoint_path),
        "eval_data": str(eval_data_path),
        "tokens_evaluated": int(tokens.shape[0]),
        "chunks": int(n_chunks),
        "seq_len": seq_len,
        "perplexity": ppl,
    }


def main():
    """CLI: python -m framework.evaluation.eval_runner."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="path to .bin file")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=100)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = evaluate_checkpoint(
        Path(args.checkpoint),
        Path(args.config),
        Path(args.data),
        seq_len=args.seq_len,
        max_chunks=args.max_chunks,
    )

    print(f"\n{'='*60}")
    print(f"PERPLEXITY: {result['perplexity']:.4f}")
    print(f"{'='*60}")
    print(f"Tokens: {result['tokens_evaluated']}")
    print(f"Chunks: {result['chunks']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
