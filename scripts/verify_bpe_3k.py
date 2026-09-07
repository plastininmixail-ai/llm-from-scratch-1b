"""Скрипт для обучения с BPE 3K словарём."""
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Используем load_with_workaround
import json
from tokenizer.bpe import BPETokenizer

t = BPETokenizer()
data = json.load(open(ROOT / "tokenizer/vocab_3k.json"))
# Manual load
t.vocab = {}
for k, v in data["vocab"].items():
    try:
        t.vocab[int(k)] = bytes.fromhex(v)
    except (ValueError, TypeError):
        t.vocab[k] = bytes.fromhex(v) if isinstance(v, str) else v
t.token_to_id = {v: k for k, v in t.vocab.items()}
t.merges = []
for merge_str in data["merges"]:
    parts = merge_str.split(" ", 1)
    if len(parts) == 2:
        a_hex, b_hex = parts
        try:
            t.merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        except ValueError:
            pass
t.merge_ranks = {pair: i for i, pair in enumerate(t.merges)}

print(f"Loaded vocab: {len(t.vocab)}")
print(f"Test 'привет': {len(t.encode('привет'))} tokens")

# Save in loadable format
import shutil
src = ROOT / "tokenizer/vocab_3k.json"
dst = ROOT / "tokenizer/vocab_3k_fixed.json"
shutil.copy(src, dst)
print(f"Copied to {dst}")
