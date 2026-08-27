# LLM From Scratch

Собственная GPT-style LLM, обученная с нуля на корпусе русской и английской Википедии + Python-код.

**Все компоненты написаны с нуля** на чистом Python / PyTorch:
- BPE-токенизатор (~250 LOC, stdlib-only)
- GPT-архитектура: Multi-Head Causal Self-Attention + Pre-LN Decoder Block (~490 LOC)
- AdamW trainer с cosine LR schedule + gradient clipping (~230 LOC)
- Inference: top-k / nucleus sampling (~190 LOC)

## Финальные результаты

| Версия | Шаги | Val loss | Лучший чекпойнт |
|--------|------|----------|-----------------|
| v1 | 2000 | 4.17 | `checkpoints/small-v1-2000steps/best.pt` |
| **v2** | 8000 | 3.7405 | `checkpoints/small-v2/best.pt` |
| **v3** | 6000 (warm-start v2) | 3.6760 | `checkpoints/small-v3/best.pt` |
| **v4** | 6000 (warm-start v3) | **🏆 3.6727** | `checkpoints/small-v4/best.pt` |

**Общий прогресс**: 4.17 → **3.6727** = ↓0.50 (↓12%) за 22000 шагов warm-start + cold-start

## Метрики

- **Perplexity** (eval на 5 MB train corpus): **57.88** — **в ×26 лучше** случайной модели (vocab=1500)
- **Throughput**: до **3.2M tok/s** на CPU (3.3M модель)
- **Размер модели**: 3.3M параметров (d_model=256, n_layers=4, n_heads=4, vocab=1500)

## Структура проекта

```
llm-from-scratch/
├── data/                       # датасеты и утилиты обработки
│   ├── extract_text.py         # парсер Wikipedia XML → plain text
│   ├── merge_corpus.py         # объединение EN + RU + code
│   ├── fetch_code.py           # скачивание Python-кода с HuggingFace
│   ├── analyze_artifacts.py    # анализ артефактов в очищенном тексте
│   ├── show_samples.py         # sanity-check корпуса
│   └── raw/                    # исходные данные (не в Git)
│       ├── enwiki-50mb.xml.bz2     # 50 МБ сырой англ. Wiki
│       ├── ruwiki-50mb.xml.bz2     # 50 МБ сырой рус. Wiki
│       ├── enwiki.txt              # 53 МБ очищенный EN
│       ├── ruwiki.txt              # 89 МБ очищенный RU
│       ├── code_python.txt         # 5.4 МБ Python-код
│       └── corpus.txt              # 144.8 МБ объединённый
├── tokenizer/                  # BPE-токенизатор
│   ├── bpe.py                  # ~250 LOC, обучение/сохранение/загрузка
│   ├── train_bpe.py            # CLI для обучения
│   └── vocab.json              # обученный словарь (1500 токенов)
├── model/                      # GPT-архитектура
│   ├── config.py               # ModelConfig + YAML-загрузчик
│   ├── attention.py            # CausalSelfAttention
│   ├── mlp.py                  # FeedForward (GELU)
│   ├── layers.py               # DecoderBlock (Pre-LN)
│   └── transformer.py          # GPT главный класс + generate()
├── training/                   # обучение
│   ├── dataset.py              # TextDataset
│   ├── trainer.py              # AdamW + cosine LR + grad clip + checkpoints
│   └── train.py                # CLI
├── inference/                  # генерация
│   └── generate.py             # load_model + top-k/top-p sampling
├── configs/small.yaml          # гиперпараметры модели
├── tests/                      # pytest (30 тестов, все проходят)
├── checkpoints/                # чекпойнты (вне Git, 252 МБ)
├── logs/                       # логи (вне Git)
├── .venv/                      # Python venv (вне Git)
└── data/raw/                   # большие файлы (вне Git)
```

## Модель

- **3.28M параметров** (без учёта shared head embedding)
- **Архитектура**: decoder-only transformer, Pre-LN
  - `d_model=256`, `n_heads=4`, `n_layers=4`
  - `max_seq_len=512`, `vocab=1500`
  - FFN hidden = 4 × d_model, GELU
  - Weight tying (input/output embeddings)
  - Causal mask в self-attention
- **Конфиг**: `configs/small.yaml`
- **Тесты**: 13 unit-тестов архитектуры (causal mask, forward shape, num parameters, generate)

## Корпус

- **144.8 МБ чистого текста**, ~21k статей
  - 67k EN Wiki (50 МБ исходных → 53 МБ очищенных)
  - 7.5k RU Wiki (50 МБ → 89 МБ)
  - 963 Python-файла (5.4 МБ)
- Очистка Wiki: убираем шаблоны `{{...}}`, refs `<ref>...</ref>`, внешние ссылки, wiki-разметку
- **Токенизация**: 144 МБ → 52M токенов за ~80 секунд

## Обучение

```bash
# Обучение v2 (лучший результат)
python -m training.train \
  --config configs/small.yaml \
  --tokenizer tokenizer/vocab.json \
  --corpus data/raw/corpus.txt \
  --out-dir checkpoints/small-v2 \
  --max-steps 8000 \
  --batch-size 8 \
  --seq-len 128 \
  --lr 2e-4 \
  --warmup 400
```

| Параметр | v1 | v2 |
|----------|------|------|
| batch_size | 8 | 8 |
| seq_len | 128 | 128 |
| lr | 3e-4 | 2e-4 |
| max_steps | 2000 | 8000 |
| Время | 4.2 мин | **17 мин** |
| Peak tok/s | 1.31M | **2.64M** |

**Loss кривая (v2):**

| Step | Train | Val |
|------|-------|-----|
| 500 | 5.32 | 5.13 |
| 1000 | 4.64 | 4.54 |
| 2000 | 4.27 | 4.19 |
| 3000 | 4.11 | 4.05 |
| 3500 | 4.07 | 4.00 ← first time <4.0 |
| 4000 | 4.04 | 3.95 |
| 6000 | 3.92 | 3.81 |
| 7000 | 3.88 | 3.77 |
| 7500 | 3.86 | 3.75 |
| **8000** | **3.85** | **🏆 3.74** |

## Генерация

```bash
# С лучшим чекпойнтом v2
python -m inference.generate \
  --checkpoint checkpoints/small-v4/best.pt \
  --tokenizer tokenizer/vocab.json \
  --prompt "The history of Russia begins" \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-k 40
```

CLI-параметры:
- `--max-new-tokens`: длина генерации (default 64)
- `--temperature`: 1.0=нейтрально, <1=детерминированно, >1=креативно (default 0.8)
- `--top-k`: оставить топ-k токенов (default 50)
- `--top-p`: nucleus sampling (опционально)

## Установка / воспроизведение

```bash
# 1. Создать venv (если ещё не)
python -m venv .venv
.venv\Scripts\python -m pip install torch numpy tqdm datasets pytest

# 2. Скачать Wiki-дампы (если data/raw/ пустая)
#    или использовать существующие

# 3. Обучить BPE-токенизатор (или использовать готовый vocab.json)
python -m tokenizer.train_bpe \
  --input data/raw/corpus.txt \
  --output tokenizer/vocab.json \
  --vocab-size 1500 --limit-bytes 3145728

# 4. Обучить модель
python -m training.train --config configs/small.yaml ...

# 5. Генерировать
python -m inference.generate --checkpoint checkpoints/small-v4/best.pt \
```

## Тесты

```bash
.venv\Scripts\python -m pytest tests/ -v
# 30 passed in 1.07s
```

## Известные ограничения

- **3.3M параметров — слишком мало** для качественного текста. Нужна модель 10–100M+
- **8000 шагов** на 144 МБ корпуса — для полноценного обучения нужны миллионы шагов и миллиарды токенов
- **CPU-only**: без GPU. На GPU обучение было бы в 10–50× быстрее
- **BPE на 3 МБ**: для лучшего покрытия стоит обучить на 30+ МБ корпуса
- **Слова в генерации часто выдуманные** — это следствие маленького размера модели
- **Качество улучшается**: v1 (val 4.17) → v2 (val 3.74), loss всё ещё падает → потенциал для дальнейшего обучения

## Git-история

```
3f2b64c Step 3: training pipeline + inference
12f8946 Step 2 final: BPE vocab=1500, 4.3min on 3MB corpus
810ef9b Step 3: GPT architecture (causal mask, attention, MLP)
0ea5d94 Step 2: BPE tokenizer (~200 LOC, stdlib-only)
f7913b7 Step 1.6: improved wiki cleaner, Python code
a07884c Step 1.5: merge EN+RU corpus, 158k articles, 139.5 MB
b898529 Step 1: venv setup, wiki XML extractor
2ec7e6d Initial commit: project structure + README
```