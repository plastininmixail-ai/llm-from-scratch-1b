"""Анализ результатов диагностики: сравнение v82, v93, v94."""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
INPUT = ROOT / "data/diagnostic_results.jsonl"

GARBAGE_PATTERNS = [
    r'\bкакиш', r'\bзвол', r'\bпоменнощи', r'\bколжи', r'\bбом',
    r'\bпоме\b', r'\bдиша\b', r'\bможи\b', r'\bпуверт',
    r'\bсамтранс\b', r'\bколам\b', r'\bпусть\b.*\bконт\b',
    r'\bExct\b', r'\bsolutput\b',
    r'\d+\.\s+[А-Я]',  # нумерованные списки
    r'- +',  # bullet lists
]
GARBAGE_RE = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


def main():
    by_model = defaultdict(list)
    with INPUT.open('r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            by_model[d['model']].append(d)

    print(f"📊 Загружено: {sum(len(v) for v in by_model.values())} ответов\n")

    # Глобальная статистика по моделям
    print("=" * 80)
    print("ГЛОБАЛЬНАЯ СТАТИСТИКА")
    print("=" * 80)
    for model in ["v82", "v93", "v94"]:
        if model not in by_model:
            continue
        items = by_model[model]
        total = len(items)
        avg_len = sum(len(d['response']) for d in items) / total
        empty = sum(1 for d in items if len(d['response']) < 5)
        has_garbage = sum(1 for d in items if any(rx.search(d['response']) for rx in GARBAGE_RE))
        has_q_prefix = sum(1 for d in items if d['response'].lstrip().startswith(("Q:", "A:")))
        has_list = sum(1 for d in items if any(line.lstrip().startswith(("-", "*", "1.", "2.", "3.")) for line in d['response'].split('\n')))
        print(f"\n{model} ({total} ответов):")
        print(f"  avg length: {avg_len:.0f} chars")
        print(f"  empty (<5 chars): {empty} ({empty/total*100:.0f}%)")
        print(f"  contains garbage: {has_garbage} ({has_garbage/total*100:.0f}%)")
        print(f"  starts with Q: or A:: {has_q_prefix} ({has_q_prefix/total*100:.0f}%)")
        print(f"  contains lists: {has_list} ({has_list/total*100:.0f}%)")

    # По категориям
    print("\n" + "=" * 80)
    print("ПО КАТЕГОРИЯМ (logic — самая важная)")
    print("=" * 80)
    categories = ["logic_ru", "logic_en", "facts_ru", "facts_en", "dialog_ru", "dialog_en", "language_ru", "language_en"]
    for cat in categories:
        print(f"\n--- {cat} ---")
        for model in ["v82", "v93", "v94"]:
            if model not in by_model:
                continue
            items = [d for d in by_model[model] if d['category'] == cat]
            if not items:
                continue
            total = len(items)
            garbage = sum(1 for d in items if any(rx.search(d['response']) for rx in GARBAGE_RE))
            print(f"  {model}: {total} q, garbage={garbage} ({garbage/total*100:.0f}%)")

    # Примеры хороших/плохих ответов
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ОТВЕТОВ (logic_ru — Сколько рук у человека?)")
    print("=" * 80)
    for model in ["v82", "v93", "v94"]:
        items = [d for d in by_model[model] if d['category'] == 'logic_ru' and 'рук' in d['prompt']]
        if items:
            print(f"\n{model}: {items[0]['prompt']}")
            print(f"  → {items[0]['response'][:200]}")


if __name__ == '__main__':
    main()
