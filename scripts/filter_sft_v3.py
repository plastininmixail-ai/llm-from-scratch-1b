"""Фильтрует sft_v2.jsonl по языку — оставляет только EN/RU пары."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "data/sft_v2.jsonl"
DST = ROOT / "data/sft_v3_en_ru.jsonl"


def is_english_or_russian(text: str) -> bool:
    """True если текст EN или RU."""
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or 'ё' <= c <= 'ё' or 'Ё' <= c <= 'Ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')

    # Spanish/Portuguese markers
    spanish_markers = ['¿', '¡', 'ñ', 'ó', 'á', 'é', 'í', 'ú']
    for m in spanish_markers:
        if m in text:
            return False

    # Cyrillic OR Latin (но не смесь)
    if cyr > 0 and lat == 0:
        return True  # Russian
    if lat > 0 and cyr == 0:
        return True  # English
    return False


def main():
    n_total = 0
    n_kept = 0
    n_skipped = 0

    with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            d = json.loads(line)
            prompt = d.get("prompt", "")
            response = d.get("response", "")

            if is_english_or_russian(prompt) and is_english_or_russian(response):
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                n_skipped += 1

    print(f"📊 Фильтрация завершена:")
    print(f"   Всего: {n_total}")
    print(f"   ✅ Оставлено EN/RU: {n_kept} ({n_kept/n_total*100:.0f}%)")
    print(f"   ❌ Удалено (ES/PT/other): {n_skipped}")
    print(f"\n📤 {DST}")


if __name__ == "__main__":
    main()
