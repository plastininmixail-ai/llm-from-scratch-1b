"""SFT v95: warm-start v94 на качественных данных (Gold Dataset v2).

Улучшает v94 без увеличения размера.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Проверяем данные
DATA = ROOT / "data/sft_v2.jsonl"
if not DATA.exists():
    print(f"ERROR: {DATA} не найден")
    sys.exit(1)

# Считаем пары
n_pairs = sum(1 for _ in DATA.open(encoding="utf-8"))
print(f"📊 sft_v2.jsonl: {n_pairs} пар")

# Показываем sample
with DATA.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        d = json.loads(line)
        print(f"\n[{i+1}] prompt: {d.get('prompt', '')[:100]}")
        print(f"    response: {d.get('response', '')[:100]}")

print("\n" + "=" * 60)
print("🚀 Запуск SFT v95")
print("=" * 60)
print(f"Base: checkpoints/chat-xlarge-v94-sft-final/best.pt")
print(f"Data: {DATA}")
print(f"LR: 5e-6 (низкий для стабильности)")
print(f"Steps: 1500 (~3-4 часа на CPU)")
