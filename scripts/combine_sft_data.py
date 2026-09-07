"""Объединяет все SFT данные в один файл для обучения.

Источники:
- sft_v3_en_ru.jsonl (4,471) — фильтрованные EN/RU
- oasst1_en_ru.jsonl (45,822) — OASST1 фильтрованный
- chat_facts.jsonl (94) — стоические факты
- wiki_summaries.jsonl (10,979) — Wikipedia

Целевой файл: data/sft_combined.jsonl
"""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUT = ROOT / "data/sft_combined.jsonl"

SOURCES = [
    ROOT / "data/sft_v3_en_ru.jsonl",
    ROOT / "data/oasst1_en_ru.jsonl",
    ROOT / "data/chat_facts.jsonl",
    ROOT / "data/wiki_summaries.jsonl",
]

# Garbage markers
GARBAGE = [
    "какиш", "поме то на", "as we can be", "since the",
    "name_", "i'm sorry", "i cannot", "as an ai",
]


def is_clean(prompt: str, response: str) -> bool:
    """Проверяет качество пары."""
    if not prompt or not response:
        return False
    if len(prompt) < 10 or len(response) < 10:
        return False
    if len(prompt) > 2000 or len(response) > 2000:
        return False
    pl = prompt.lower()
    rl = response.lower()
    for m in GARBAGE:
        if m in pl or m in rl:
            return False
    return True


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n_total = 0
    n_kept = 0
    n_lang_skipped = 0
    n_quality_skipped = 0
    n_dup_skipped = 0

    sources_count = {}

    with OUT.open("w", encoding="utf-8") as out_f:
        for src in SOURCES:
            if not src.exists():
                continue
            n_src = 0
            n_src_kept = 0
            with src.open(encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    n_total += 1
                    n_src += 1
                    d = json.loads(line)
                    prompt = d.get("prompt", d.get("question", "")).strip()
                    response = d.get("response", d.get("answer", "")).strip()

                    if not is_clean(prompt, response):
                        n_quality_skipped += 1
                        continue

                    # Дубликаты
                    key = prompt[:100]
                    if key in seen:
                        n_dup_skipped += 1
                        continue
                    seen.add(key)

                    out_f.write(json.dumps({
                        "prompt": prompt,
                        "response": response,
                        "source": src.name,
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                    n_src_kept += 1

            sources_count[src.name] = (n_src, n_src_kept)
            print(f"  {src.name}: {n_src} → {n_src_kept}")

    print(f"\n📊 Итог:")
    print(f"  Всего: {n_total}")
    print(f"  ✅ Сохранено: {n_kept}")
    print(f"  ❌ Качество: {n_quality_skipped}")
    print(f"  ❌ Дубликаты: {n_dup_skipped}")
    print(f"\n📤 {OUT}")


if __name__ == "__main__":
    main()
