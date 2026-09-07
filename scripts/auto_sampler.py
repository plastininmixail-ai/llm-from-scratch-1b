"""AutoSampler: мониторит новые чекпойнты, генерирует текст, сохраняет в лог."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
from framework.inference import generate, load_model_for_inference
from framework.data.simple_tokenizer import SimpleBPETokenizer


CKPT_DIR = Path("checkpoints/v1_fineweb")
CONFIG = Path("configs/v0_1b.yaml")
TOKENIZER = Path("tokenizer/vocab_3k.json")
HISTORY = Path("logs/sample_history.jsonl")
LOG = Path("logs/sampler.log")

LOG.parent.mkdir(parents=True, exist_ok=True)

# Разнообразные тестовые промпты
PROMPTS = [
    "The capital of France is",
    "Machine learning is",
    "Once upon a time",
    "Python is a programming language",
    "The meaning of life is",
    "In the beginning",
    "Science is",
    "The history of",
]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.rstrip())


def get_sampled_steps():
    if not HISTORY.exists():
        return set()
    sampled = set()
    for line in HISTORY.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            row = json.loads(line)
            sampled.add(row["checkpoint"])
        except Exception:
            pass
    return sampled


def sample_checkpoint(ckpt_path: Path) -> list:
    """Генерирует примеры для всех промптов."""
    log(f"Loading {ckpt_path.name}...")
    model = load_model_for_inference(ckpt_path, CONFIG)
    tok = SimpleBPETokenizer(str(TOKENIZER))

    results = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        if not ids:
            ids = [1]
        x = torch.tensor([ids], dtype=torch.long)
        out = generate(model, x, max_new_tokens=30, temperature=0.7, top_k=20)
        gen_ids = out[0, len(ids):].tolist()
        gen_text = tok.decode(gen_ids)
        results.append({"prompt": prompt, "generation": gen_text})
        log(f"  {prompt!r} -> {gen_text!r}")

    return results


def main():
    log(f"=== AutoSampler started ===")
    log(f"Watching: {CKPT_DIR}")
    log(f"Prompts: {len(PROMPTS)}")
    log(f"Already sampled: {len(get_sampled_steps())} checkpoints")

    while True:
        sampled = get_sampled_steps()

        ckpts = sorted(
            [p for p in CKPT_DIR.glob("step*.pt") if p.is_file()],
            key=lambda p: int(p.stem.replace("step", ""))
        )

        new_ckpts = [c for c in ckpts if c.name not in sampled]

        if new_ckpts:
            log(f"Found {len(new_ckpts)} new checkpoints: {[c.name for c in new_ckpts]}")
            for ckpt in new_ckpts:
                try:
                    samples = sample_checkpoint(ckpt)
                    record = {
                        "checkpoint": ckpt.name,
                        "step": int(ckpt.stem.replace("step", "")),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "samples": samples,
                    }
                    with HISTORY.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log(f"  ✓ Saved {len(samples)} samples")
                except Exception as e:
                    log(f"  ERROR on {ckpt.name}: {e}")
        else:
            log(f"No new checkpoints (have {len(ckpts)} total)")

        log(f"Sleeping 10 min...")
        time.sleep(600)


if __name__ == "__main__":
    main()
