"""Токенизация jsonl → binary (uint16)."""
import json
import sys
from pathlib import Path

import numpy as np

# Используем существующий BPE (но нам нужен новый с vocab=32000)
sys.path.insert(0, ".")
from tokenizer.bpe import BPETokenizer


def tokenize_jsonl(input_path: Path, output_path: Path, tokenizer: BPETokenizer, max_bytes: int = 0):
    """Токенизирует jsonl → uint16 binary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_ids = []
    total_bytes = 0

    with input_path.open(encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if max_bytes and total_bytes > max_bytes:
                break
            try:
                item = json.loads(line)
                # Prompt + response
                prompt = item.get("prompt", "").strip()
                response = item.get("response", "").strip()
                if prompt and response:
                    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
                    ids = tokenizer.encode(text)
                    all_ids.extend(ids)
                    total_bytes += len(line)
            except Exception:
                pass

    if not all_ids:
        print(f"No data from {input_path}")
        return 0

    # Save as uint16
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"  {input_path.name}: {len(arr)/1e6:.2f}M tokens -> {output_path}")
    return len(arr)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory with jsonl files")
    p.add_argument("--output-dir", required=True, help="Output directory for .bin files")
    p.add_argument("--tokenizer", required=True, help="Tokenizer vocab.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-bytes", type=int, default=0)
    args = p.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Tokenize each jsonl
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        out_path = output_dir / (jsonl_path.stem + ".bin")
        n = tokenize_jsonl(jsonl_path, out_path, tokenizer, args.max_bytes)
        total += n

    print(f"\nTotal tokens: {total/1e6:.2f}M")


if __name__ == "__main__":
    main()
