"""Анализ RU vs EN на v82 и v94 — одна и та же тема, два языка."""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
INPUT = ROOT / "data/ru_en_results.jsonl"

# Расширенные garbage patterns
GARBAGE_PATTERNS = [
    r'\bкакиш', r'\bзвол', r'\bпоменнощи', r'\bколжи', r'\bбом',
    r'\bпоме\b', r'\bдиша\b', r'\bможи\b', r'\bпуверт',
    r'\bсамтранс\b', r'\bколам\b',
    r'\bExct\b', r'\bsolutput\b', r'\bputput\b',
    r'\d+\.\s+[А-ЯA-Z]',  # нумерованные списки
    r'^[A-Z][a-z]+:[A-Z]',  # Q: ... или A: ...
    r'\bExampl', r'\bexample\b.*\b:',  # шаблонные ответы
]
GARBAGE_RE = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


def is_garbage(text: str) -> bool:
    if not text or len(text) < 5:
        return True
    if any(rx.search(text) for rx in GARBAGE_RE):
        return True
    return False


def is_readable(text: str) -> bool:
    """Ответ читаемый: не пустой, не garbage, не слишком короткий."""
    if not text or len(text) < 10:
        return False
    if is_garbage(text):
        return False
    # Проверяем что есть хоть какие-то слова
    words = re.findall(r'\w+', text)
    return len(words) >= 3


def main():
    by_model_lang = defaultdict(list)
    with INPUT.open('r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            key = (d['model'], d['lang'])
            by_model_lang[key].append(d)

    print("=" * 80)
    print("📊 RU vs EN СРАВНЕНИЕ v82 vs v94")
    print("=" * 80)

    print("\n📈 Readable ответы (%):")
    print(f"  {'Модель':<8} {'RU':<10} {'EN':<10} {'Разница EN-RU'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*15}")
    for model in ['v82', 'v94']:
        ru_items = by_model_lang.get((model, 'ru'), [])
        en_items = by_model_lang.get((model, 'en'), [])
        ru_readable = sum(1 for d in ru_items if is_readable(d['response']))
        en_readable = sum(1 for d in en_items if is_readable(d['response']))
        ru_pct = ru_readable / max(len(ru_items), 1) * 100
        en_pct = en_readable / max(len(en_items), 1) * 100
        diff = en_pct - ru_pct
        sign = '+' if diff > 0 else ''
        print(f"  {model:<8} {ru_pct:.0f}% ({ru_readable}/{len(ru_items)})  {en_pct:.0f}% ({en_readable}/{len(en_items)})    {sign}{diff:.0f}%")

    print("\n📉 Garbage ответы (%):")
    print(f"  {'Модель':<8} {'RU':<10} {'EN':<10} {'Разница RU-EN'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*15}")
    for model in ['v82', 'v94']:
        ru_items = by_model_lang.get((model, 'ru'), [])
        en_items = by_model_lang.get((model, 'en'), [])
        ru_garbage = sum(1 for d in ru_items if is_garbage(d['response']))
        en_garbage = sum(1 for d in en_items if is_garbage(d['response']))
        ru_pct = ru_garbage / max(len(ru_items), 1) * 100
        en_pct = en_garbage / max(len(en_items), 1) * 100
        diff = ru_pct - en_pct
        sign = '+' if diff > 0 else ''
        print(f"  {model:<8} {ru_pct:.0f}% ({ru_garbage}/{len(ru_items)})  {en_pct:.0f}% ({en_garbage}/{len(en_items)})    {sign}{diff:.0f}%")

    # По категориям (logic, facts, dialog, abstract, lifestyle)
    print("\n" + "=" * 80)
    print("📊 ПО КАТЕГОРИЯМ (logic — самая показательная)")
    print("=" * 80)
    print(f"  {'Модель':<8} {'Категория':<12} {'RU':<8} {'EN':<8} {'Разница'}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

    # Нужно восстановить категории — у нас их нет в данных, но можно по pair_id разбить
    # PAIRS: 1-10 facts, 11-20 logic, 21-30 dialog, 31-40 abstract, 41-50 lifestyle
    def get_category(pair_id):
        if pair_id <= 10:
            return "facts"
        elif pair_id <= 20:
            return "logic"
        elif pair_id <= 30:
            return "dialog"
        elif pair_id <= 40:
            return "abstract"
        else:
            return "lifestyle"

    categories = ["logic", "facts", "dialog", "abstract", "lifestyle"]
    for model in ['v82', 'v94']:
        for cat in categories:
            ru_items = [d for d in by_model_lang.get((model, 'ru'), []) if get_category(d['pair_id']) == cat]
            en_items = [d for d in by_model_lang.get((model, 'en'), []) if get_category(d['pair_id']) == cat]
            if not ru_items or not en_items:
                continue
            ru_readable = sum(1 for d in ru_items if is_readable(d['response']))
            en_readable = sum(1 for d in en_items if is_readable(d['response']))
            ru_pct = ru_readable / len(ru_items) * 100
            en_pct = en_readable / len(en_items) * 100
            diff = en_pct - ru_pct
            sign = '+' if diff > 0 else ''
            print(f"  {model:<8} {cat:<12} {ru_pct:.0f}%     {en_pct:.0f}%     {sign}{diff:.0f}%")

    # Парные примеры — одна тема, RU vs EN
    print("\n" + "=" * 80)
    print("📝 ПРИМЕРЫ ПАРНЫХ ОТВЕТОВ (одна тема, RU vs EN)")
    print("=" * 80)

    sample_pairs = [1, 11, 21, 31, 41]  # по одному из каждой категории
    for pair_id in sample_pairs:
        ru_d = next((d for d in by_model_lang.get(('v82', 'ru'), []) if d['pair_id'] == pair_id), None)
        en_d = next((d for d in by_model_lang.get(('v82', 'en'), []) if d['pair_id'] == pair_id), None)
        if ru_d and en_d:
            print(f"\n--- Pair #{pair_id} ({get_category(pair_id)}) ---")
            print(f"  RU: {ru_d['prompt']}")
            print(f"  RU ответ: {ru_d['response'][:200]}")
            print(f"  EN: {en_d['prompt']}")
            print(f"  EN ответ: {en_d['response'][:200]}")


if __name__ == '__main__':
    main()
