"""Тест inference на текущем лучшем чекпойнте."""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import torch
from framework.inference import generate, load_model_for_inference
from framework.data.simple_tokenizer import SimpleBPETokenizer

CHECKPOINT = Path("checkpoints/v0_1b_1000/step200.pt")
CONFIG = Path("configs/v0_1b.yaml")
TOKENIZER = Path("tokenizer/vocab_3k.json")

print("Loading model...")
model = load_model_for_inference(CHECKPOINT, CONFIG)
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

print("Loading tokenizer...")
tok = SimpleBPETokenizer(TOKENIZER)

prompts = [
    "The capital of France is",
    "Привет, как дела?",
    "Machine learning is",
    "Python is a programming language",
    "Once upon a time",
]

print("\n" + "=" * 60)
print("GENERATION TEST")
print("=" * 60)

for prompt in prompts:
    try:
        ids = tok.encode(prompt)
        if not ids:
            ids = [1, 2, 3]
            prompt_display = "[no tokens]"
        else:
            prompt_display = prompt

        x = torch.tensor([ids], dtype=torch.long)
        out = generate(model, x, max_new_tokens=30, temperature=0.7, top_k=20)
        gen_ids = out[0, len(ids):].tolist()
        gen_text = tok.decode(gen_ids)

        print(f"\n[PROMPT] {prompt_display}")
        print(f"[GEN]    {gen_text}")
    except Exception as e:
        print(f"\n[ERROR] {prompt}: {e}")

print("\n" + "=" * 60)
