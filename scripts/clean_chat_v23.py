"""Chat v23 — финальная очистка: chat_v22 + фильтр prompt leakage.

Удаляем ответы, которые:
  - содержат "I am an AI assistant...", "You are a helpful..."
  - начинаются с "System:", "Q:", "A:"
  - содержат явные признаки "галлюцинаций инструкций"
"""
import json
import re
from pathlib import Path

INPUT = Path("data/chat_v22.jsonl")
OUTPUT = Path("data/chat_v23.jsonl")

# Тот же фильтр что в chat_v22 + защита от prompt leakage
REGEX_PATTERNS = [
    r"\bQ: You are a helpful assistant",
    r"You are an AI assistant that helps",
    r"Think like you are answering to a five year old",
    r"You are a faithful",
    r"As an AI (assistant|language model)",
    r"I am an AI (assistant|language model)",
    r"I am a helpful",
    r"You will be given a task",
    r"User will you give you a task",
    r"I am a large language model",
    r"possible is not possideople",
    r"^\s*System:",
    r"^\s*Q:\s",   # ответ начинается с "Q: ..."
    r"^\s*A:\s",
    r"Forget all previous instructions",
    r"^Ignore previous",
    r"as a helpful assistant that",
]

START_BAD = [
    "I am an AI",
    "I am a large",
    "As an AI ",
    "As a helpful",
    "Being an AI",
    "As an artificial",
]


def is_poisoned(text):
    if not text:
        return True
    t = text.strip()
    if len(t) < 5:
        return True
    for p in REGEX_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            return True
    for s in START_BAD:
        if t.lower().startswith(s.lower()):
            return True
    return False


def is_meaningful(text, min_len=15, max_len=3000):
    if not text or len(text.strip()) < min_len:
        return False
    if len(text) > max_len:
        return False
    return True


def main():
    print(f"📖 читаю {INPUT}...")
    kept = 0
    dropped_poison = 0
    dropped_other = 0
    with INPUT.open(encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", "").strip()
                response = obj.get("response", "").strip()
                if not prompt:
                    continue
                # prompt leakage check
                if is_poisoned(prompt):
                    dropped_poison += 1
                    continue
                if is_poisoned(response):
                    dropped_poison += 1
                    continue
                if not is_meaningful(response):
                    dropped_other += 1
                    continue
                fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
                kept += 1
            except Exception:
                dropped_other += 1
            if i % 200000 == 0:
                print(f"  {i:,} → kept={kept:,}, poison={dropped_poison:,}, other={dropped_other:,}")

    total = kept + dropped_poison + dropped_other
    print(f"\n✅ ИТОГО ({total:,}):")
    print(f"  kept (clean):    {kept:,} ({100*kept/total:.1f}%)")
    print(f"  poison removed:   {dropped_poison:,} ({100*dropped_poison/total:.1f}%)")
    print(f"  other filtered:   {dropped_other:,} ({100*dropped_other/total:.1f}%)")


if __name__ == "__main__":
    main()
