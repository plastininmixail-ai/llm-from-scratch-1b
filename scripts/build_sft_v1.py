"""Создание SFT-датасета для правильного обучения.

Источники:
  - OASST1 (49,529 пар) — OpenAssistant диалоги (качественные, человеческие)
  - НЕ используем chat_v23 (он вызвал мусор в base model)

Целевой размер: 10,000-20,000 пар (микс RU + EN для устойчивости).

Шаги:
  1. Фильтруем OASST1: длина промпта 10-500, ответа 20-1000, без markdown/code
  2. Делим на RU/EN по кириллице
  3. Балансируем: ~50% RU, ~50% EN
  4. Сохраняем в data/sft_v1.jsonl
"""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OASST1 = ROOT / "data/oasst1_clean.jsonl"
OUTPUT = ROOT / "data/sft_v1.jsonl"


def has_cyrillic(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or 'ё' <= c <= 'ё' or 'Ё' <= c <= 'Ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    return cyr > lat * 0.3  # 30% кириллицы = русский


CODE_MARKERS = ['\ndef ', '\nclass ', '\nimport ', '\nfrom ', '\n    def ', '```', '|}|',
                '```python', '```cpp', '```javascript', '{', '}', '()(', '[][']
MARKDOWN = ['##', '###', '**', '__', '![', '](http']
GARBAGE = ['lol', 'XD', 'asdf', 'qwerty', 'lol xd', '^^']


def is_clean(prompt: str, response: str) -> bool:
    if not prompt or not response:
        return False
    if len(prompt) < 10 or len(prompt) > 500:
        return False
    if len(response) < 20 or len(response) > 1500:
        return False
    # Слишком много markdown — не диалог
    md_count = sum(prompt.count(m) + response.count(m) for m in MARKDOWN)
    if md_count > 5:
        return False
    # Код
    for m in CODE_MARKERS:
        if m in response:
            return False
    # Мусор
    for g in GARBAGE:
        if g in prompt.lower() or g in response.lower():
            return False
    # Много спецсимволов подряд
    if re.search(r'([\.\,\-\=]{5,})', response):
        return False
    # Ссылки на wikipedia/ref
    if 'wikipedia.org' in prompt.lower() or 'wikipedia.org' in response.lower():
        return False
    if '<ref>' in response or '<sup>' in response:
        return False
    return True


def main():
    if not OASST1.exists():
        print(f"❌ {OASST1} не найден")
        return

    print(f"📖 читаю {OASST1}...")

    ru_pairs = []
    en_pairs = []
    seen = set()

    with OASST1.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                p = obj.get('prompt', '').strip()
                r = obj.get('response', '').strip()
                if not is_clean(p, r):
                    continue
                key = (p[:100], r[:100])
                if key in seen:
                    continue
                seen.add(key)
                if has_cyrillic(p + r):
                    ru_pairs.append({'prompt': p, 'response': r})
                else:
                    en_pairs.append({'prompt': p, 'response': r})
            except Exception:
                continue

    print(f"✅ собрано: RU={len(ru_pairs)}, EN={len(en_pairs)}")

    # Балансируем: возьмём минимум из двух сторон × 2 (чтобы было 50/50)
    target = min(len(ru_pairs), len(en_pairs)) * 2
    target_ru = min(len(ru_pairs), target // 2)
    target_en = min(len(en_pairs), target // 2)

    print(f"📊 target: RU={target_ru}, EN={target_en}, total={target_ru + target_en}")

    final = ru_pairs[:target_ru] + en_pairs[:target_en]
    print(f"📤 пишу {len(final)} пар в {OUTPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"✅ Готово: {len(final)} пар сохранено")


if __name__ == '__main__':
    main()
