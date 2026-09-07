"""Конвертирует SFT данные в ChatML формат для лучшего SFT.

ChatML format:
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
"""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "data/sft_combined.jsonl"
DST = ROOT / "data/sft_chatml.jsonl"


def to_chatml(prompt: str, response: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response.strip()}<|im_end|>"
    )


count = 0
with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
    for line in fin:
        try:
            item = json.loads(line)
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            if prompt and response:
                text = to_chatml(prompt, response)
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
        except Exception:
            pass

print(f"✅ {count} ChatML пар в {DST}")
