"""Тест v101 (новый vocab 3000 + seq 512)."""
import sys
from pathlib import Path
import json

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Patch load
from tokenizer import bpe as bpe_mod
def _patched_load(cls, path):
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
bpe_mod.BPETokenizer.load = classmethod(_patched_load)

from inference.generate import generate_text, load_model_from_checkpoint
import time

print("Loading v101 (vocab 3000, seq 512)...")
model, tokenizer, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v101-3k-seq512/best.pt",
    ROOT / "tokenizer/vocab_3k.json"
)
print("✓ Loaded\n")

QUESTIONS = [
    ("RU", "Привет!"),
    ("RU", "Что такое гравитация?"),
    ("RU", "Расскажи про космос"),
    ("RU", "Кто такой Эйнштейн?"),
    ("RU", "Что такое нейронная сеть?"),
    ("EN", "Hello!"),
    ("EN", "What is machine learning?"),
    ("EN", "Tell me about Einstein"),
]

print("=" * 70)
print("v101 TEST (vocab 3000 + seq 512)")
print("=" * 70)

for lang, q in QUESTIONS:
    print(f"\n[{lang}] Q: {q}")
    t0 = time.time()
    try:
        r = generate_text(
            model, tokenizer, prompt=q,
            max_new_tokens=80, temperature=0.3,
            top_k=40, top_p=0.95, repetition_penalty=1.3,
            no_repeat_ngram_size=4, device="cpu"
        )
        r = r.strip()[:200]
    except Exception as e:
        r = f"[ERR: {e}]"
    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s] A: {r}")
