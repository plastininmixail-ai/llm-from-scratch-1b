"""Универсальный sampler: загрузить любой чекпойнт и сгенерировать текст."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, ".")

import torch
from framework.inference import generate, load_model_for_inference
from framework.data.simple_tokenizer import SimpleBPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/v0_1b.yaml")
    p.add_argument("--tokenizer", default="tokenizer/vocab_3k.json")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=3)
    args = p.parse_args()

    print(f"Loading {args.checkpoint}...")
    model = load_model_for_inference(Path(args.checkpoint), Path(args.config))
    tok = SimpleBPETokenizer(args.tokenizer)
    print(f"OK ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    print(f"\nPrompt: {args.prompt!r}")
    print("=" * 60)

    ids = tok.encode(args.prompt)
    if not ids:
        ids = [1]
    x = torch.tensor([ids], dtype=torch.long)

    for i in range(args.n_samples):
        out = generate(model, x, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k)
        gen_ids = out[0, len(ids):].tolist()
        gen_text = tok.decode(gen_ids)
        print(f"[{i+1}] {gen_text}")

    print("=" * 60)


if __name__ == "__main__":
    main()
