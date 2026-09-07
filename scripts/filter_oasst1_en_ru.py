"""Фильтрует oasst1_clean по языку → oasst1_en_ru.jsonl."""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "data/oasst1_clean.jsonl"
DST = ROOT / "data/oasst1_en_ru.jsonl"


def is_english(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'ё' <= c <= 'ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z')
    return lat > cyr * 2


def is_russian(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'ё' <= c <= 'ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z')
    return cyr > lat * 2


def is_quality(d: dict) -> bool:
    """Качественная пара."""
    prompt = d.get("prompt", "").strip()
    response = d.get("response", "").strip()
    if len(prompt) < 20 or len(response) < 20:
        return False
    if len(prompt) > 1500 or len(response) > 1500:
        return False
    if "I'm sorry" in response or "I cannot" in response:
        return False
    return True


def main():
    n_total = 0
    n_kept = 0
    n_skipped_quality = 0
    n_skipped_lang = 0

    with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            d = json.loads(line)
            text = d.get("prompt", "") + d.get("response", "")

            if not is_quality(d):
                n_skipped_quality += 1
                continue

            if is_russian(text) or is_english(text):
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                n_skipped_lang += 1

    print(f"OASST1 clean: {n_total}")
    print(f"  ✅ Сохранено EN/RU: {n_kept}")
    print(f"  ❌ Пропущено по качеству: {n_skipped_quality}")
    print(f"  ❌ Пропущено по языку: {n_skipped_lang}")
    print(f"\n📤 {DST}")


if __name__ == "__main__":
    main()
