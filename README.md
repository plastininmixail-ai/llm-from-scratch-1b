# 🚀 Bridge Bot — Руководство

## Что это
Telegram бот на собственной LLM модели.

## Архитектура (v15)

```
Пользователь → Telegram API
    ↓
Bridge Bot (inference/bridge_bot.py)
    ├── Intent Classifier (intent_classifier.py, 389 примеров)
    ├── Sentiment Analyzer (sentiment.py, 8 эмоций)
    ├── Empathy prefix
    ↓
[1] Hardcoded (1461 паттернов) → 90%+ покрытие
    ↓ (если не найдено)
[2] TF-IDF KB (11,939 записей) → быстрый поиск
    ↓
[3] Web Agent v5 (если свежая инфа)
[4] Code Agent v3 (если вычисления)
    ↓
[5] v103 model (113M, RMSNorm + RoPE)
    ↓
Ответ пользователю
```

## Как запустить

```bash
# Установить зависимости
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Запустить Bridge Bot
.venv/Scripts/python.exe -m inference.bridge_bot \
  --checkpoint checkpoints/chat-xlarge-v103-15k/best.pt \
  --tokenizer tokenizer/vocab_3k.json
```

## Чекпойнты

| Run | val | Когда использовать |
|-----|-----|-------------------|
| **v103** | **4.49** | Лучший (рекомендую) |
| v94 | 3.62 | Запасной (Bridge Bot v14) |

## Файлы

- `PLAN.md` — план работы
- `docs/WEEK1_SUMMARY.md` — итоги недели
- `HARD_BANS.md` — запреты (стоицизм, force-kill)
- `inference/bridge_bot.py` — главный файл бота
- `model/transformer.py` — GPT с RMSNorm + RoPE
- `model/rmsnorm.py` — RMSNorm
- `model/rope.py` — RoPE

## Известные проблемы

1. **113M мало** для связного русского текста
2. **vocab 3000** — нужен 16-32K
3. **CPU обучение** медленное
4. **Модели мусорят** на свободные темы (но hardcoded + KB покрывает 90%)
