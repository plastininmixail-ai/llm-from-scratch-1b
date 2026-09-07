"""Eval benchmark: считает perplexity на разных типах данных.

Запускает perplexity на:
1. Wiki (factual text)
2. SFT data (instruction style)
3. OpenStax (educational)
4. Random tokens (sanity check)

Сравнивает разные чекпойнты.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, ".")

import torch
from framework.inference import load_model_for_inference
from framework.evaluation import compute_perplexity


CHECKPOINTS = sorted(
    [p for p in Path("checkpoints").rglob("best.pt") if p.is_file()],
    key=lambda p: p.stat().st_mtime
)

EVAL_FILES = {
    "wiki": "data/binary/v0/wiki_summaries_full.bin",
    "openstax": "data/binary/v0/openstax_textbooks.bin",
    "chat": "data/binary/v0/chat_v23.bin",
    "sft": "data/binary/v0/sft_combined.bin",
}

OUTPUT = Path("logs/eval_benchmark.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def load_tokens(path: Path, max_tokens: int = 50 * 512) -> torch.Tensor:
    if not path.exists():
        return None
    tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    n = min(max_tokens, len(tokens))
    return torch.from_numpy(tokens[:n])


def main():
    if not CHECKPOINTS:
        print("No checkpoints found")
        return

    print(f"Found {len(CHECKPOINTS)} checkpoints:")
    for ckpt in CHECKPOINTS:
        print(f"  {ckpt}")

    print(f"\nEvaluating on {len(EVAL_FILES)} datasets...")
    print("=" * 80)

    results = []
    for ckpt_path in CHECKPOINTS:
        print(f"\nLoading {ckpt_path.name}...")
        try:
            model = load_model_for_inference(ckpt_path, Path("configs/v0_1b.yaml"))
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        ckpt_result = {"checkpoint": str(ckpt_path), "timestamp": datetime.now().isoformat(), "perplexities": {}}

        for name, file_path in EVAL_FILES.items():
            tokens = load_tokens(Path(file_path))
            if tokens is None or len(tokens) < 100:
                ckpt_result["perplexities"][name] = None
                continue

            try:
                ppl = compute_perplexity(model, tokens, seq_len=512)
                ckpt_result["perplexities"][name] = round(ppl, 4)
                print(f"  {name:10s}: PPL = {ppl:.4f}")
            except Exception as e:
                print(f"  {name}: ERROR {e}")
                ckpt_result["perplexities"][name] = None

        results.append(ckpt_result)

        # Save after each checkpoint
        with OUTPUT.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 80)
    print("Saved to", OUTPUT)

    # Print summary table
    print("\nSummary:")
    print(f"{'Checkpoint':<40} | {'Wiki':>10} | {'OpenStax':>10} | {'Chat':>10} | {'SFT':>10}")
    print("-" * 100)
    for r in results:
        ckpt = Path(r["checkpoint"]).parent.name
        ppls = r["perplexities"]
        row = f"{ckpt:<40}"
        for name in EVAL_FILES:
            v = ppls.get(name)
            row += f" | {v if v else 'N/A':>10}"
        print(row)


if __name__ == "__main__":
    main()
