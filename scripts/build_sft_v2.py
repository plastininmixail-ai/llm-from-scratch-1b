"""Создание финального SFT v2 датасета: sft_v1 (OASST1) + gold_v2 (qwen3).

Цель: 7,580 OASST1 + до 5,000 qwen3 = 12,580+ пар максимум.

Каждый промпт встречается только один раз (дедупликация).
"""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SFT_V1 = ROOT / "data/sft_v1.jsonl"
GOLD_V2 = ROOT / "data/gold_dataset_v2.jsonl"
OUTPUT = ROOT / "data/sft_v2.jsonl"


def main():
    pairs = []
    seen = set()

    for path in [SFT_V1, GOLD_V2]:
        if not path.exists():
            print(f"⚠ {path} не найден, пропускаю")
            continue
        n = 0
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    p = obj.get('prompt', '').strip()
                    r = obj.get('response', '').strip()
                    if not p or not r:
                        continue
                    key = (p[:100], r[:100])
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({
                        "prompt": p,
                        "response": r,
                        "source": path.stem,
                    })
                    n += 1
                except Exception:
                    continue
        print(f"📖 {path.name}: {n} пар")

    print(f"\n📊 Всего уникальных: {len(pairs)}")

    # Перемешиваем чтобы OASST1 и gold_v2 шли вперемешку
    import random
    random.seed(42)
    random.shuffle(pairs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"📤 {OUTPUT}")


if __name__ == '__main__':
    main()
