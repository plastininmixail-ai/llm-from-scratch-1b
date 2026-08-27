"""
Eval perplexity модели на стандартных бенчмарках.

Использование:
    python -m inference.eval_perplexity \
        --checkpoint checkpoints/small-v2/best.pt \
        --tokenizer tokenizer/vocab.json \
        --config configs/small.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from inference.generate import load_model_from_checkpoint
from training.dataset import TextDataset


@torch.no_grad()
def evaluate_perplexity(
    model,
    dataset: TextDataset,
    device: str = "cpu",
    batch_size: int = 4,
    max_iters: int | None = None,
) -> dict:
    """
    Считает perplexity = exp(mean loss) на датасете.

    Returns: {"loss": float, "perplexity": float, "tokens": int}
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if max_iters is not None and i >= max_iters:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        # F.cross_entropy с reduction='sum' чтобы получить сумму loss по токенам
        # logits: (B, T, vocab); y: (B, T)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        n_tokens = y.numel()
        total_loss += loss.item()
        total_tokens += n_tokens
        n_batches += 1

    mean_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(mean_loss)
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "batches": n_batches,
    }


def eval_on_corpus(
    name: str,
    text: str,
    tokenizer,
    seq_len: int,
    model,
    device: str,
    max_iters: int | None = None,
) -> dict:
    """Считает perplexity на тексте."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        tmp_path = f.name

    ds = TextDataset(tmp_path, tokenizer, seq_len=seq_len, stride=seq_len)
    Path(tmp_path).unlink()
    result = evaluate_perplexity(
        model, ds, device=device, batch_size=4, max_iters=max_iters
    )
    result["name"] = name
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="опционально, иначе восстановим из state_dict")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-iters", type=int, default=None,
                   help="ограничить число батчей на бенчмарк")
    args = p.parse_args()

    print(f"загружаю чекпойнт {args.checkpoint}…")
    model, tok, cfg = load_model_from_checkpoint(
        args.checkpoint, args.tokenizer, device=args.device
    )
    print(f"модель: {model.info()}")
    print(f"vocab: {len(tok)}")
    print()

    results = []

    # 1. Наш обучающий корпус (in-domain)
    print("[eval 1] train corpus (in-domain, первые 5 МБ)…")
    train_text = Path("data/raw/corpus.txt").read_text(encoding="utf-8")[:5_000_000]
    r = eval_on_corpus(
        "train_corpus (5MB head)",
        train_text, tok, cfg.max_seq_len, model, args.device,
        max_iters=args.max_iters,
    )
    results.append(r)
    print(f"  loss={r['loss']:.4f}  perplexity={r['perplexity']:.2f}  tokens={r['tokens']:,}")
    print()

    # 2. WikiText-103 (out-of-domain test)
    print("[eval 2] WikiText-103 (out-of-domain)…")
    try:
        from datasets import load_dataset
        wt = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="test",
            streaming=False, trust_remote_code=True,
        )
        wt_text = "\n\n".join([t for t in wt["text"] if t.strip()])[:2_000_000]
        del wt
        r = eval_on_corpus(
            "wikitext-103 (2MB head)",
            wt_text, tok, cfg.max_seq_len, model, args.device,
            max_iters=args.max_iters or 200,
        )
        results.append(r)
        print(f"  loss={r['loss']:.4f}  perplexity={r['perplexity']:.2f}  tokens={r['tokens']:,}")
    except Exception as e:
        print(f"  пропускаем: {e}")
    print()

    # 3. Synthetic toy corpus (для sanity check)
    print("[eval 3] синтетический повторяющийся текст (sanity)…")
    toy = ("The quick brown fox jumps over the lazy dog. " * 10000)
    r = eval_on_corpus(
        "synthetic_repeat",
        toy, tok, cfg.max_seq_len, model, args.device,
        max_iters=50,
    )
    results.append(r)
    print(f"  loss={r['loss']:.4f}  perplexity={r['perplexity']:.2f}  tokens={r['tokens']:,}")
    print()

    # 4. Random noise (нижняя граница перплексии)
    print("[eval 4] случайный шум (lower bound)…")
    import random
    random.seed(42)
    noise = " ".join([random.choice(["a", "b", "c", "d"]) for _ in range(100000)])
    r = eval_on_corpus(
        "random_noise",
        noise, tok, cfg.max_seq_len, model, args.device,
        max_iters=50,
    )
    results.append(r)
    print(f"  loss={r['loss']:.4f}  perplexity={r['perplexity']:.2f}  tokens={r['tokens']:,}")
    print()

    # Сводная таблица
    print("=" * 70)
    print(f"{'Корпус':<30} {'Loss':>10} {'Perplexity':>15} {'Tokens':>12}")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['loss']:>10.4f} {r['perplexity']:>15.2f} {r['tokens']:>12,}")

    print()
    print("Сравнение с известными моделями (perplexity на WikiText-103):")
    print("  - GPT-2 (117M):       ~29.4")
    print("  - GPT-2 (345M):       ~22.8")
    print("  - GPT-2 (774M):       ~18.5")
    print("  - трансформер с нуля 3.3M: ожидаемо 500–2000 (модель маленькая)")
    print("  - random baseline:    ~vocab_size = 1500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())