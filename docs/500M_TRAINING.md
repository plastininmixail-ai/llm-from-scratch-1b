# 500M Model Training Pipeline

## Архитектура

- **d_model**: 1280
- **n_heads**: 20
- **n_layers**: 32
- **vocab_size**: 1500 (BPE)
- **Параметры**: ~500M
- **max_seq_len**: 256

## 3 фазы обучения

### Phase 1: Pre-training (4-6 часов)

Данные: `data/raw/chat_xlarge_corpus.txt`

```bash
python -m training.train \
  --config configs/xxxlarge.yaml \
  --tokenizer tokenizer/vocab.json \
  --corpus data/raw/chat_xlarge_corpus.txt \
  --max-steps 5000 \
  --batch-size 1 \
  --seq-len 256 \
  --lr 2e-4 \
  --warmup 500 \
  --out-dir checkpoints/chat-xxxlarge-pretrain
```

### Phase 2: SFT (2-3 часа)

Данные: `data/sft_chatml.jsonl` (25,687 ChatML пар)

```bash
python -m training.train \
  --config configs/xxxlarge.yaml \
  --tokenizer tokenizer/vocab.json \
  --corpus data/sft_chatml.jsonl \
  --max-steps 2000 \
  --batch-size 1 \
  --seq-len 256 \
  --lr 5e-6 \
  --warmup 200 \
  --base checkpoints/chat-xxxlarge-pretrain/best.pt \
  --reset-step \
  --out-dir checkpoints/chat-xxxlarge-sft
```

### Phase 3: DPO (5-10 минут)

Данные: `data/dpo_dataset.jsonl` (4500 пар)

```bash
python -m training.dpo_train \
  --base checkpoints/chat-xxxlarge-sft/best.pt \
  --data data/dpo_dataset.jsonl \
  --max-pairs 1000 \
  --lr 1e-6 \
  --beta 0.1
```

## Или всё сразу:

```bash
python scripts/train_500m.py
```

## Ожидаемые результаты

| Phase | val_loss | Качество |
|-------|----------|----------|
| 1 | 3.5-4.0 | Базовый язык |
| 2 | 2.8-3.2 | Лучше SFT |
| 3 | 0.01-0.5 | DPO |

## Требования

- CPU: 32+ GB RAM (8+ cores)
- Диск: 50+ GB для чекпойнтов
- Время: 7-10 часов на все фазы

## Ограничения

- vocab_size=1500 (текущий BPE) — может быть мало для 500M
- max_seq_len=256 — для длинных текстов нужно увеличить
- CPU обучение — медленно (GPU в 5-10 раз быстрее)

## Альтернативы

- Использовать существующую 302M модель (xxlarge) — быстрее
- Обучать только SFT фазу (без pre-training) — 2-3 часа
- Уменьшить размер до 250M — 3-4 часа
