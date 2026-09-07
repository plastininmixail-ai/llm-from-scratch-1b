"""Подготовка к обучению 500M модели.

Создаёт:
1. Конфиг для 500M модели
2. Скрипт обучения (3 фазы)
3. Документацию
"""
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

# 1. Конфиг 500M
CONFIG_500M = """# 500M model config — Bridge Bot 500M
# Создано: 2026-09-05
# Оценка параметров: 500M

model:
  d_model: 1280
  n_heads: 20
  n_layers: 32
  max_seq_len: 256
  vocab_size: 1500
  mlp_hidden_mult: 4
  tie_weights: true

training:
  batch_size: 1
  seq_len: 256
  lr: 2e-4
  warmup: 500
  weight_decay: 0.01
  grad_clip: 1.0

# Phase 1: Pre-training
pretrain:
  data: data/raw/chat_xlarge_corpus.txt
  max_steps: 5000
  estimated_time: 4-6 hours

# Phase 2: SFT
sft:
  data: data/sft_chatml.jsonl  # 25,687 ChatML пар
  max_steps: 2000
  lr: 5e-6
  estimated_time: 2-3 hours

# Phase 3: DPO
dpo:
  data: data/dpo_dataset.jsonl  # 4500 пар
  epochs: 1
  lr: 1e-6
  estimated_time: 5-10 min
"""

(ROOT / "configs/xxxlarge.yaml").write_text(CONFIG_500M, encoding="utf-8")
print(f"✅ Created {ROOT}/configs/xxxlarge.yaml")

# 2. Скрипт обучения 500M
TRAIN_SCRIPT = '''"""Train 500M Bridge Bot.

3 phases:
1. Pre-training на corpus
2. SFT на ChatML данных
3. DPO на preference данных
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")


def run_phase(phase_name: str, cmd: list[str]):
    print(f"\\n{'='*70}")
    print(f"🚀 Phase: {phase_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"❌ Phase {phase_name} failed with code {result.returncode}")
        sys.exit(1)
    print(f"✅ Phase {phase_name} completed")


def main():
    # Phase 1: Pre-training
    run_phase("Pre-training 500M", [
        "python", "-m", "training.train",
        "--config", "configs/xxxlarge.yaml",
        "--tokenizer", "tokenizer/vocab.json",
        "--corpus", "data/raw/chat_xlarge_corpus.txt",
        "--max-steps", "5000",
        "--batch-size", "1",
        "--seq-len", "256",
        "--lr", "2e-4",
        "--warmup", "500",
        "--out-dir", "checkpoints/chat-xxxlarge-pretrain",
    ])

    # Phase 2: SFT
    run_phase("SFT 500M", [
        "python", "-m", "training.train",
        "--config", "configs/xxxlarge.yaml",
        "--tokenizer", "tokenizer/vocab.json",
        "--corpus", "data/sft_chatml.jsonl",
        "--max-steps", "2000",
        "--batch-size", "1",
        "--seq-len", "256",
        "--lr", "5e-6",
        "--warmup", "200",
        "--base", "checkpoints/chat-xxxlarge-pretrain/best.pt",
        "--reset-step",
        "--out-dir", "checkpoints/chat-xxxlarge-sft",
    ])

    # Phase 3: DPO
    run_phase("DPO 500M", [
        "python", "-m", "training.dpo_train",
        "--base", "checkpoints/chat-xxxlarge-sft/best.pt",
        "--data", "data/dpo_dataset.jsonl",
        "--max-pairs", "1000",
        "--lr", "1e-6",
        "--beta", "0.1",
    ])

    print("\\n" + "="*70)
    print("🎉 All 3 phases completed!")
    print("="*70)


if __name__ == "__main__":
    main()
'''

(ROOT / "scripts/train_500m.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
print(f"✅ Created {ROOT}/scripts/train_500m.py")

# 3. Документация
DOC = """# 500M Model Training Pipeline

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
python -m training.train \\
  --config configs/xxxlarge.yaml \\
  --tokenizer tokenizer/vocab.json \\
  --corpus data/raw/chat_xlarge_corpus.txt \\
  --max-steps 5000 \\
  --batch-size 1 \\
  --seq-len 256 \\
  --lr 2e-4 \\
  --warmup 500 \\
  --out-dir checkpoints/chat-xxxlarge-pretrain
```

### Phase 2: SFT (2-3 часа)

Данные: `data/sft_chatml.jsonl` (25,687 ChatML пар)

```bash
python -m training.train \\
  --config configs/xxxlarge.yaml \\
  --tokenizer tokenizer/vocab.json \\
  --corpus data/sft_chatml.jsonl \\
  --max-steps 2000 \\
  --batch-size 1 \\
  --seq-len 256 \\
  --lr 5e-6 \\
  --warmup 200 \\
  --base checkpoints/chat-xxxlarge-pretrain/best.pt \\
  --reset-step \\
  --out-dir checkpoints/chat-xxxlarge-sft
```

### Phase 3: DPO (5-10 минут)

Данные: `data/dpo_dataset.jsonl` (4500 пар)

```bash
python -m training.dpo_train \\
  --base checkpoints/chat-xxxlarge-sft/best.pt \\
  --data data/dpo_dataset.jsonl \\
  --max-pairs 1000 \\
  --lr 1e-6 \\
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
"""

(ROOT / "docs/500M_TRAINING.md").parent.mkdir(parents=True, exist_ok=True)
(ROOT / "docs/500M_TRAINING.md").write_text(DOC, encoding="utf-8")
print(f"✅ Created {ROOT}/docs/500M_TRAINING.md")

print("\n" + "="*70)
print("🎉 Все файлы для 500M созданы!")
print("="*70)
