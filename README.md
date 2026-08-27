# LLM From Scratch

Собственная GPT-style LLM, обученная с нуля на корпусе русской и английской Википедии.

## Структура проекта

```
llm-from-scratch/
├── data/              # датасеты
├── tokenizer/         # свой BPE-токенизатор
├── model/             # архитектура трансформера
├── training/          # пайплайн обучения
├── inference/         # генерация текста
├── checkpoints/       # сохранённые веса (вне Git)
├── logs/              # метрики обучения (вне Git)
├── configs/           # гиперпараметры
├── notebooks/         # эксперименты
└── tests/             # тесты
```

## Стек

- Python 3.12+
- PyTorch (CPU)
- Свой BPE-токенизатор (без библиотек)
- Своя архитектура (без transformers/llama-index)

## Этапы

1. ✅ Подготовка среды, скачивание корпуса
2. ⬜ BPE-токенизатор с нуля
3. ⬜ Архитектура трансформера
4. ⬜ Обучение (CPU, маленькая модель ~25M)
5. ⬜ Генерация текста
6. ⬜ Сохранение/загрузка весов
7. ⬜ Визуализация и оценка

## Как начать

```bash
cd ~/Desktop/llm-from-scratch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```