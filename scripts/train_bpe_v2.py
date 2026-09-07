"""Обучение BPE на качественном корпусе для Stage 0/1."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

OUT_PATH = Path("tokenizer/vocab_8k.json")
CORPUS = "data/raw/big_corpus.txt"

# Собираем все данные в один файл
import glob
print("Finding source files...")
sources = []
for pattern in [
    "data/sources/wiki_api/*.json",
    "data/sources/openstax_pdfs/*.json",
    "data/sources/fastai/**/*.txt",
    "data/sources/d2l/**/*.txt",
    "data/wiki_summaries_full.jsonl",
    "data/openstax_textbooks.jsonl",
    "data/sources/sep/*.html",
]:
    sources.extend(glob.glob(pattern, recursive=True))

print(f"Found {len(sources)} source files")

# Пишем в один файл
OUT = Path("data/raw/corpus_combined.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

n_chars = 0
n_docs = 0
with OUT.open("w", encoding="utf-8") as f:
    for src in sources:
        try:
            p = Path(src)
            if p.suffix == ".json":
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k in ("text", "content", "article"):
                        if k in data:
                            f.write(str(data[k]) + "\n\n")
                            n_chars += len(str(data[k]))
                            n_docs += 1
                            break
                    else:
                        for v in data.values():
                            if isinstance(v, str):
                                f.write(v + "\n\n")
                                n_chars += len(v)
                                n_docs += 1
                                break
            elif p.suffix == ".jsonl":
                import json
                with p.open(encoding="utf-8") as fj:
                    for line in fj:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                text = row.get("text", row.get("content", row.get("response", "")))
                                f.write(str(text) + "\n\n")
                                n_chars += len(str(text))
                                n_docs += 1
                        except Exception:
                            pass
            elif p.suffix in (".html", ".txt", ".md"):
                text = p.read_text(encoding="utf-8", errors="ignore")
                f.write(text + "\n\n")
                n_chars += len(text)
                n_docs += 1
        except Exception as e:
            print(f"  Skip {src}: {e}")

print(f"\nWrote {n_docs} docs, {n_chars/1e6:.1f}M chars to {OUT}")
print(f"Now training BPE vocab=8000...")

from tokenizer.train_bpe import train_bpe
import json as json_mod
data = json_mod.load(open("tokenizer/train_bpe.py".replace("train_bpe.py","train_bpe.py")))
