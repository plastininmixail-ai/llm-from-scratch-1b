"""Конвертировать FineWeb-Edu parquet → text → binary для обучения."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow not installed. Install with: pip install pyarrow")
    sys.exit(1)

SRC = Path("data/sources/fineweb_edu/datasets--HuggingFaceFW--fineweb-edu/snapshots")
OUT_TXT = Path("data/raw/fineweb_edu_text.txt")
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(SRC.rglob("sample/10BT/*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

n_docs = 0
n_chars = 0
with OUT_TXT.open("w", encoding="utf-8") as out:
    for i, pf in enumerate(parquet_files):
        print(f"[{i+1}/{len(parquet_files)}] {pf.name} ({pf.stat().st_size/1e6:.0f} MB)")
        try:
            table = pq.read_table(pf, columns=["text"])
            for text in table["text"]:
                s = str(text)
                if len(s) > 50:  # фильтр слишком коротких
                    out.write(s + "\n\n")
                    n_docs += 1
                    n_chars += len(s)
                    if n_docs % 50000 == 0:
                        print(f"  {n_docs} docs, {n_chars/1e6:.1f}M chars")
        except Exception as e:
            print(f"  ERROR: {e}")

print(f"\nTotal: {n_docs} docs, {n_chars/1e6:.1f}M chars")
print(f"Saved to {OUT_TXT}")
