"""Добавить MetaMathQA в SFT данные для Stage 1 reasoning training."""
import sys
import json
from pathlib import Path
sys.path.insert(0, ".")


SFT_PATH = Path("data/sft_stage1/sft_raw.jsonl")
OUT_PATH = Path("data/sft_stage1/sft_with_math.jsonl")

print("=== Adding MetaMathQA ===\n")

# MetaMathQA доступен через HuggingFace
try:
    from datasets import load_dataset
    print("Downloading MetaMathQA (sample)...")
    # Используем sample чтобы не качать 400MB
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    print("OK loaded streaming")
except Exception as e:
    print(f"HF error: {e}")
    print("Trying alternative...")
    try:
        ds = load_dataset("Skywork/MetaMathQA", split="train", streaming=True)
        print("OK via Skywork")
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        ds = None

added = 0
out = OUT_PATH.open("w", encoding="utf-8")

# Сохраняем существующие
if SFT_PATH.exists():
    print(f"\nReading existing {SFT_PATH}...")
    for line in SFT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.write(line + "\n")
    existing = sum(1 for _ in SFT_PATH.open())
    print(f"  Existing: {existing} pairs")
else:
    existing = 0

# Добавляем MetaMathQA
if ds is not None:
    print(f"\nAdding MetaMathQA samples...")
    for i, row in enumerate(ds):
        if i >= 10000:  # лимит для Stage 1
            break
        try:
            # MetaMathQA schema
            question = row.get("query") or row.get("question") or row.get("instruction")
            answer = row.get("response") or row.get("answer") or row.get("output")
            if question and answer:
                # Filter quality
                if len(question) < 10 or len(answer) < 50:
                    continue
                if len(question) > 1000 or len(answer) > 3000:
                    continue

                pair = {
                    "prompt": str(question).strip(),
                    "response": str(answer).strip(),
                    "source": "metamathqa",
                }
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                added += 1

                if added % 1000 == 0:
                    print(f"  Added {added}...")
        except Exception as e:
            continue

out.close()
print(f"\nTotal MetaMathQA added: {added}")
print(f"Output: {OUT_PATH}")

# Stats
total = sum(1 for _ in OUT_PATH.open(encoding="utf-8"))
print(f"Total pairs in sft_with_math: {total}")
