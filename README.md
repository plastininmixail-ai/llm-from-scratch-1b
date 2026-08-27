# LLM From Scratch

Собственная GPT-style LLM, обученная с нуля на корпусе русской и английской Википедии + Python-код из GitHub.

## Структура проекта

```
llm-from-scratch/
├── data/              # даки и скрипты их подготовки
├── tokenizer/         # свой BPE-токенизатор (~200 LOC, stdlib-only)
├── model/             # GPT-style decoder-only трансформер (~300 LOC)
├── training/          # пайплайн обучения (следующий шаг)
├── inference/         # генерация текста
├── checkpoints/       # сохранённые веса (вне Git)
├── logs/              # метрики (вне Git)
├── configs/           # YAML-конфиги моделей
├── tests/             # pytest'ы
└── tokenizer/vocab.json  # обученный BPE (после шага 2)
```

## Стек

- Python 3.12+, PyTorch 2.13 (CPU)
- Только stdlib для токенизатора (re, json, collections)
- Своя архитектура (без transformers/llama-index)

## Прогресс

| Шаг | Что сделано |
|-----|-------------|
| 1 ✅ | Окружение, скачивание корпуса (144.8 МБ Wiki EN+RU + 5 МБ Python-кода) |
| 2 ✅ | BPE-токенизатор с нуля, 17 тестов проходят. Обучение на полном корпусе (32 МБ subset для скорости) |
| 3 ✅ | GPT-style архитектура: CausalSelfAttention, FeedForward, DecoderBlock (Pre-LN), GPT main. 30 тестов проходят |
| 4 ⬜ | Обучение модели (next) |
| 5 ⬜ | Генерация текста, evaluation |
| 6 ⬜ | Сохранение/загрузка вечей |

## Архитектура (Шаг 3)

- **Embedding**: vocab_size × d_model + позиционные embeddings
- **DecoderBlock** (× N): Pre-LN → Causal MHA → Pre-LN → FFN(GELU) → residual
- **Causal MHA**: Q,K,V через fused Linear, scaled dot-product с верхнетреугольной маской -inf
- **LM Head**: weight-tied с tok_emb
- **Размер**: ~10.8M параметров (d_model=384, n_heads=6, n_layers=6, vocab=8000)

## Как начать

```bash
cd ~/Desktop/llm-from-scratch
.venv\Scripts\activate
pip install -r requirements.txt

# Запустить тесты
pytest tests/ -v

# Обучить токенизатор (30–90 минут)
python -m tokenizer.train_bpe --input data/raw/corpus.txt --output tokenizer/vocab.json --vocab-size 8000
```

## Конфиги

- `configs/small.yaml` — 10.8M параметров, для CPU-обучения (~2–4 ч/эпоха на 140 МБ)
- Для большей модели: `d_model=512, n_layers=8` (~50M параметров)

## Документы в корпусе

- **16 305** EN Wikipedia статей (49 МБ)
- **7 505** RU Wikipedia статей (90 МБ)
- **963** Python-файла с GitHub (5.4 МБ)
- **Итого**: 21 782 документа, 144.8 МБ (после shuffle)

## Git-история

```
810ef9b Step 3: GPT architecture
0ea5d94 Step 2: BPE tokenizer
f7913b7 Step 1.6: improved wiki cleaner, Python code
a07884c Step 1.5: merge EN+RU corpus
b898529 Step 1: venv setup, wiki extractor
2ec7e6d Initial commit: structure + README
```