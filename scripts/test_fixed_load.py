"""Запускает обучение v101 с vocab 3000 + seq_len 512."""
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Patch load для BPETokenizer
import json
from tokenizer import bpe as bpe_mod

original_load = bpe_mod.BPETokenizer.load

def patched_load(cls, path):
    """Workaround: load vocabulary properly."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    t = cls()
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
    return t

bpe_mod.BPETokenizer.load = classmethod(patched_load)

# Теперь импортируем generate и обучаем
from inference.generate import load_model_from_checkpoint
print("Testing load with fixed vocab...")
tokenizer = bpe_mod.BPETokenizer.load(ROOT / "tokenizer/vocab_3k.json")
print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")

# Кодируем "привет мир"
ids = tokenizer.encode("Привет мир")
print(f"'Привет мир' → {ids} ({len(ids)} tokens)")
