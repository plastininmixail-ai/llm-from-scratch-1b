"""Eval watcher: ждёт новые чекпойнты и считает perplexity."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch

from framework.inference import load_model_for_inference
from framework.evaluation import compute_perplexity


CKPT_DIR = Path("checkpoints/v1_fineweb")
CONFIG = Path("configs/v0_1b.yaml")
EVAL_DATA = Path("data/binary/v0/wiki_summaries_full.bin")
HISTORY = Path("logs/eval_history.jsonl")
LOG = Path("logs/eval_watcher.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
HISTORY.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.rstrip())


def get_evaluated_steps():
    """Возвращает set шагов, которые уже оценены."""
    if not HISTORY.exists():
        return set()
    evaluated = set()
    for line in HISTORY.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            row = json.loads(line)
            evaluated.add(row["checkpoint"])
        except Exception:
            pass
    return evaluated


def eval_checkpoint(ckpt_path: Path) -> dict:
    """Оценка одного чекпойнта."""
    log(f"Loading {ckpt_path.name}...")
    model = load_model_for_inference(ckpt_path, CONFIG)

    if not EVAL_DATA.exists():
        log(f"  Skip: {EVAL_DATA} not found")
        return None

    # Загружаем eval данные
    tokens = torch.from_numpy(
        __import__("numpy").fromfile(EVAL_DATA, dtype="uint16").astype("int64")
    )
    # Берём первые 50 chunks × 512 токенов
    n_tokens = min(50 * 512, tokens.shape[0])
    tokens = tokens[:n_tokens]

    log(f"  Computing perplexity on {n_tokens} tokens...")
    ppl = compute_perplexity(model, tokens, seq_len=512)

    result = {
        "checkpoint": ckpt_path.name,
        "step": int(ckpt_path.stem.replace("step", "")),
        "perplexity": ppl,
        "tokens_evaluated": int(n_tokens),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log(f"  ✓ {ckpt_path.name}: PPL={ppl:.4f}")
    return result


def main():
    log(f"=== Eval watcher started ===")
    log(f"Watching: {CKPT_DIR}")
    log(f"Already evaluated: {len(get_evaluated_steps())} checkpoints")

    while True:
        evaluated = get_evaluated_steps()

        # Ищем новые чекпойнты
        ckpts = sorted(
            [p for p in CKPT_DIR.glob("step*.pt") if p.is_file()],
            key=lambda p: int(p.stem.replace("step", ""))
        )

        new_ckpts = [c for c in ckpts if c.name not in evaluated]

        if new_ckpts:
            log(f"Found {len(new_ckpts)} new checkpoints: {[c.name for c in new_ckpts]}")
            for ckpt in new_ckpts:
                try:
                    result = eval_checkpoint(ckpt)
                    if result:
                        with HISTORY.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(result) + "\n")
                except Exception as e:
                    log(f"  ERROR on {ckpt.name}: {e}")
        else:
            log(f"No new checkpoints (have {len(ckpts)} total)")

        # Спим 10 минут
        log(f"Sleeping 10 min...")
        time.sleep(600)


if __name__ == "__main__":
    main()
