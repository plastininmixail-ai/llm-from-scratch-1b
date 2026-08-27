"""
CLI для обучения BPE-токенизатора на корпусе.

Использование:
    python -m tokenizer.train_bpe \
        --input data/raw/corpus.txt \
        --output tokenizer/vocab.json \
        --vocab-size 8000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .bpe import BPETokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="путь к .txt файлу с корпусом")
    p.add_argument("--output", type=Path, required=True,
                   help="куда сохранить vocab.json")
    p.add_argument("--vocab-size", type=int, default=8000,
                   help="итоговый размер словаря")
    p.add_argument("--min-pair-freq", type=int, default=2)
    p.add_argument("--limit-bytes", type=int, default=0,
                   help="ограничить корпус (0 = без лимита)")
    p.add_argument("--checkpoint", action="store_true",
                   help="сохранять checkpoint каждые 500 merges")
    args = p.parse_args()

    if not args.input.exists():
        print(f"нет файла: {args.input}")
        return 1

    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"vocab_size: {args.vocab_size}")

    start = time.time()

    SEP = b"\r\n\r\n"  # windows-style разделитель (как пишет Python в text mode)

    def text_iter():
        with args.input.open("rb") as f:
            buf = b""
            chars_read = 0
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    if buf.strip():
                        yield buf.decode("utf-8", errors="replace")
                    break
                buf += chunk
                chars_read += len(chunk)
                while SEP in buf:
                    doc, buf = buf.split(SEP, 1)
                    if doc.strip():
                        yield doc.decode("utf-8", errors="replace")
                if args.limit_bytes and chars_read >= args.limit_bytes:
                    break

    tok = BPETokenizer()
    tok.fit(text_iter(),
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            verbose=True,
            checkpoint_path=str(args.output) if args.checkpoint else None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.output)

    elapsed = time.time() - start
    print(f"\nготово за {elapsed:.1f} с ({elapsed/60:.1f} мин)")
    print(f"vocab: {len(tok)} tokens, merges: {len(tok.merges)}")
    print(f"сохранено: {args.output}")

    # краткая sanity-проверка
    print("\n--- sanity ---")
    for t in ["hello world", "привет мир", "def hello()"]:
        ids = tok.encode(t)
        decoded = tok.decode(ids)
        print(f"  {t!r:30} → {len(ids):3} tok → {decoded!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())