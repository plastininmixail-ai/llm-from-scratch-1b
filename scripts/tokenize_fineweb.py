"""Токенизация FineWeb-Edu text → binary для обучения."""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

INPUT = Path("data/raw/fineweb_edu_text.txt")
OUT_DIR = Path("data/binary/v1_fineweb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER = Path("tokenizer/vocab_3k.json")
MAX_BYTES = 30_000_000_000  # 30 GB target

print(f"Input: {INPUT} ({INPUT.stat().st_size/1e9:.1f} GB)")
print(f"Tokenizer: {TOKENIZER}")
print(f"Output dir: {OUT_DIR}")

# Используем тот же encoder что в framework
from framework.data.simple_tokenizer import SimpleBPETokenizer

tok = SimpleBPETokenizer(str(TOKENIZER))
print(f"Vocab: {len(tok.vocab)}")

bin_path = OUT_DIR / "fineweb_edu.bin"
manifest_path = OUT_DIR / "manifest.json"

n_tokens = 0
n_bytes = 0
file_idx = 0

with INPUT.open("r", encoding="utf-8", errors="ignore") as f:
    buffer = []
    buf_size = 0
    for chunk in iter(lambda: f.read(10_000_000), ""):  # 10 MB chunks
        if n_bytes >= MAX_BYTES:
            break

        # Токенизируем chunk
        # Делим по абзацам
        paragraphs = chunk.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue

            try:
                ids = tok.encode(para)
                if not ids:
                    continue
                # Добавляем EOS marker (используем 0 как конец документа)
                ids.append(0)
                buffer.extend(ids)
                n_tokens += len(ids)
                buf_size += len(ids) * 2  # uint16
                n_bytes += len(para.encode("utf-8"))

                # Сохраняем каждые 50M токенов
                if buf_size >= 100_000_000:
                    arr = np.array(buffer, dtype=np.uint16)
                    fp = OUT_DIR / f"fineweb_{file_idx:03d}.bin"
                    arr.tofile(fp)
                    file_idx += 1
                    print(f"  Saved {fp.name}: {len(arr)/1e6:.1f}M tokens, total {n_tokens/1e6:.1f}M")
                    buffer = []
                    buf_size = 0
            except Exception as e:
                pass

# Save leftover
if buffer:
    arr = np.array(buffer, dtype=np.uint16)
    fp = OUT_DIR / f"fineweb_{file_idx:03d}.bin"
    arr.tofile(fp)
    file_idx += 1
    print(f"  Saved {fp.name}: {len(arr)/1e6:.1f}M tokens")

# Manifest
import json
manifest = {
    "source": str(INPUT),
    "tokenizer": str(TOKENIZER),
    "files": file_idx,
    "total_tokens": int(n_tokens),
    "total_bytes_input": int(n_bytes),
}
with manifest_path.open("w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nTotal: {n_tokens/1e6:.1f}M tokens in {file_idx} files")
print(f"Manifest saved to {manifest_path}")
