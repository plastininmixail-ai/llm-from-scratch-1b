"""Pipeline для v94 SFT: OASST1 + qwen3 gold_v2.

Запускать ПОСЛЕ завершения v93 (которая сейчас тренируется).

Шаги:
  1. Сгенерировать gold_v2 через Ollama qwen3 (5,000 пар, ~30 мин)
  2. Собрать sft_v2.jsonl = OASST1 + gold_v2 (дедуплицированный)
  3. SFT v94: warm-start от v93, chat_v23 + gold_v2 (после v93 завершится)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")


def run(cmd: str, name: str):
    print(f"\n{'='*60}")
    print(f"🚀 {name}")
    print(f"{'='*60}")
    rc = subprocess.run(cmd, shell=True, cwd=ROOT).returncode
    if rc != 0:
        print(f"❌ {name} завершился с ошибкой {rc}")
        sys.exit(1)


def main():
    # Шаг 1: генерация gold_v2 через Ollama
    run("python -m scripts.gen_gold_v2", "Генерация gold_v2 через Ollama qwen3")

    # Шаг 2: сборка финального датасета
    run("python -m scripts.build_sft_v2", "Сборка sft_v2 (OASST1 + gold_v2)")

    # Шаг 3: SFT v94
    run(
        "python -m training.chat_train_xlarge "
        "--data data/sft_v2.jsonl "
        "--out-dir checkpoints/chat-xlarge-v94-sft-final "
        "--max-steps 5000 "
        "--batch-size 1 --seq-len 256 "
        "--base checkpoints/chat-xlarge-v93-sft-oasst1/best.pt "
        "--reset-step "
        "--lr 3e-6 --warmup 200",
        "SFT v94: warm-start от v93 на sft_v2"
    )

    print("\n✅ Pipeline завершён успешно!")


if __name__ == '__main__':
    main()
