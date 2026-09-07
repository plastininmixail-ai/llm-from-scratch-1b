"""Тест v104 DPO."""
import sys
import json
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

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

print("Loading v104 (DPO)...")
model, tokenizer, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v104-dpo-v2/best.pt",
    ROOT / "tokenizer/vocab_3k.json"
)
print("✓ Loaded\n")

QUESTIONS = [
    "Привет!",
    "Что такое гравитация?",
    "Расскажи про космос",
    "Кто такой Эйнштейн?",
    "Hello!",
    "What is machine learning?",
]

print("=" * 60)
print("v104 TEST (DPO trained)")
print("=" * 60)

for q in QUESTIONS:
    print(f"\nQ: {q}")
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
