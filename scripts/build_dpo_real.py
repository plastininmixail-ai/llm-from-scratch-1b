"""DPO на v103: chosen=hardcoded ответы, rejected=генерация v103 (мусор)."""
import json
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

# Генерируем chosen/rejected пары из hardcoded + реальных ответов
# chosen = правильный ответ из hardcoded
# rejected = мусорный вывод v103 (имитация)

HARDCODED_FILE = ROOT / "data/kb_misc_tools.jsonl"
HARDCODED_FILE2 = ROOT / "data/kb_openstax.jsonl"
HARDCODED_FILE3 = ROOT / "data/kb_arxiv.jsonl"

dpo_pairs = []

# Загружаем хорошие ответы из KB
for f in [HARDCODED_FILE, HARDCODED_FILE2, HARDCODED_FILE3]:
    if not f.exists():
        continue
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                p = json.loads(line)
                prompt = p.get("prompt", "").strip()
                response = p.get("response", "").strip()
                if prompt and response and len(response) > 20:
                    # rejected = имитация мусора (перевёрнутые слова + шум)
                    rejected = " ".join(response.split()[::-1])[:200]
                    dpo_pairs.append({
                        "prompt": prompt,
                        "chosen": response[:500],
                        "rejected": rejected,
                    })
                    if len(dpo_pairs) >= 300:
                        break
            except Exception:
                pass
    if len(dpo_pairs) >= 300:
        break

print(f"DPO pairs: {len(dpo_pairs)}")
OUT = ROOT / "data/dpo_real.jsonl"
OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in dpo_pairs), encoding="utf-8")
print(f"✅ Saved to {OUT}")
