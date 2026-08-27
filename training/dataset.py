"""
Загрузчик датасета для обучения LLM.

Что делает:
  1. Читает корпус (text file)
  2. Токенизирует через наш BPE-токенизатор
  3. Конкатенирует все токены в один длинный массив (BOS-разделителей нет,
     но конец статьи = конец последовательности, что естественно)
  4. Режет на окна фиксированной длины seq_len (для контекста модели)
  5. Для каждого окна (x) строит target (y) = сдвинутый на 1 вправо x:
       context:  [t0, t1, t2, ..., t_{n-1}]
       target:   [t1, t2, ..., t_{n-1}, IGNORE]
     (стандартный LM-target: предсказать следующий токен)
  6. Каждый элемент __getitem__ возвращает (x, y) как torch.long

Использование:
  tok = BPETokenizer.load("tokenizer/vocab.json")
  ds = TextDataset("data/raw/corpus.txt", tok, seq_len=512)
  x, y = ds[0]
  # x.shape == (seq_len,), y.shape == (seq_len,)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer.bpe import BPETokenizer


def encode_corpus(path: str | Path, tokenizer: BPETokenizer,
                  add_bos: bool = True) -> np.ndarray:
    """Токенизирует файл целиком, возвращает np.int32 массив токенов.

    Статьи разделяем маркером BOS (id 256, токен </w>), чтобы модель
    понимала границы текстов.
    """
    bos_id = 256  # </w> — естественный разделитель статей
    text = Path(path).read_text(encoding="utf-8")
    # разбиваем по двойному переводу строки (статьи)
    docs = [d for d in text.split("\n\n") if d.strip()]
    all_ids: list[int] = []
    for i, doc in enumerate(docs):
        ids = tokenizer.encode(doc)
        all_ids.extend(ids)
        if i < len(docs) - 1 and add_bos:
            # разделитель статей
            all_ids.append(bos_id)
    return np.array(all_ids, dtype=np.int32)


class TextDataset(Dataset):
    def __init__(self, corpus_path: str | Path, tokenizer: BPETokenizer,
                 seq_len: int = 512, stride: int | None = None):
        """
        corpus_path: путь к .txt файлу
        tokenizer: обученный BPE
        seq_len: длина контекстного окна
        stride: шаг между окнами (None = seq_len, без перекрытия)
        """
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokens = encode_corpus(corpus_path, tokenizer, add_bos=True)
        self.num_tokens = len(self.tokens)
        # каждый пример — окно seq_len + 1 (последний токен для target)
        # длина массива = num_tokens - seq_len
        self.num_examples = max(0, (self.num_tokens - seq_len) // self.stride)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y

    def info(self) -> str:
        return (f"TextDataset: {self.num_tokens:,} tokens, "
                f"seq_len={self.seq_len}, stride={self.stride}, "
                f"examples={self.num_examples:,}")