"""Анализ распределения sft_combined.jsonl по языкам и длинам."""
import json
import re
from collections import Counter
from pathlib import Path

PATH = Path("C:/Users/mixai/Desktop/llm-from-scratch/data/sft_combined.jsonl")


def is_russian(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'ё' <= c <= 'ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z')
    return cyr > lat * 2


def is_english(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'ё' <= c <= 'ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z')
    return lat > cyr * 2


def main():
    langs = Counter()
    sources = Counter()
    prompt_lens = []
    response_lens = []

    with PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d["prompt"] + d["response"]
            if is_russian(text):
                langs["ru"] += 1
            elif is_english(text):
                langs["en"] += 1
            else:
                langs["other"] += 1

            sources[d.get("source", "?")] += 1
            prompt_lens.append(len(d["prompt"]))
            response_lens.append(len(d["response"]))

    total = sum(langs.values())
    print(f"Всего: {total}")
    print(f"\nПо языкам:")
    for lang, count in langs.most_common():
        print(f"  {lang}: {count} ({count/total*100:.1f}%)")
    print(f"\nПо источникам:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")
    print(f"\nСредняя длина prompt: {sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Средняя длина response: {sum(response_lens)/len(response_lens):.0f}")


if __name__ == "__main__":
    main()
