"""Stage 1: подготовка SFT данных для instruction following."""
import sys
import json
from pathlib import Path
sys.path.insert(0, ".")

# Собираем SFT данные из разных источников
SOURCES = [
    "data/sft_combined.jsonl",     # 25K (EN+RU mix)
    "data/sft_v2.jsonl",           # 7.7K
    "data/oasst1_en_ru.jsonl",     # 45K
    "data/sft_fastai.jsonl",       # 27
    "data/sft_sources_combined.jsonl",  # 123
]

OUT_DIR = Path("data/sft_stage1")
OUT_DIR.mkdir(exist_ok=True)

print("=== Stage 1: SFT Data Collection ===\n")

total = 0
all_pairs = []

for src in SOURCES:
    p = Path(src)
    if not p.exists():
        print(f"  ✗ {src} (not found)")
        continue

    count = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Нормализуем формат {prompt, response}
                if "prompt" in row and "response" in row:
                    all_pairs.append({
                        "prompt": str(row["prompt"]),
                        "response": str(row["response"]),
                        "source": src,
                    })
                    count += 1
                elif "instruction" in row and "output" in row:
                    # OASST формат
                    all_pairs.append({
                        "prompt": str(row["instruction"]),
                        "response": str(row["output"]),
                        "source": src,
                    })
                    count += 1
            except Exception:
                pass

    total += count
    print(f"  ✓ {src}: {count} pairs")

print(f"\nTotal: {total} pairs")

# Quality filter: удалить мусор
filtered = []
for pair in all_pairs:
    p = pair["prompt"].strip()
    r = pair["response"].strip()

    # Пропускаем слишком короткие или длинные
    if len(p) < 5 or len(r) < 10:
        continue
    if len(p) > 2000 or len(r) > 4000:
        continue

    # Пропускаем пустые / мусор
    if not any(c.isalnum() for c in p):
        continue
    if not any(c.isalnum() for c in r):
        continue

    filtered.append(pair)

print(f"After quality filter: {len(filtered)} pairs")

# Save
out_path = OUT_DIR / "sft_raw.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for pair in filtered:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"Saved to {out_path}")

# Stats
en_count = sum(1 for p in filtered if all(ord(c) < 128 for c in p["prompt"] + p["response"]))
ru_count = sum(1 for p in filtered if any(0x0400 <= ord(c) <= 0x04FF for c in p["prompt"] + p["response"]))

print(f"\nLanguage distribution:")
print(f"  EN-only: {en_count}")
print(f"  With RU: {ru_count}")
