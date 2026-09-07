"""Свой Dataset для binary файлов (uint16 токены)."""
import json
from pathlib import Path

import numpy as np
import torch


class BinaryDataset:
    """Читает .bin файлы с uint16 токенами."""

    def __init__(self, data_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.data_path = Path(data_path)

        # Находим .bin файлы
        self.files = sorted(self.data_path.glob("*.bin"))
        if not self.files:
            # Fallback — читаем .jsonl файлы как есть
            self.files = []
            self.mode = "jsonl"
            self.jsonl_files = sorted(self.data_path.glob("*.jsonl"))
            if not self.jsonl_files:
                # Ищем выше
                self.jsonl_files = sorted(Path("data").glob("*.jsonl"))
        else:
            self.mode = "binary"

        if self.mode == "binary":
            # Memory-map binary files
            self.data_arrays = []
            for f in self.files:
                arr = np.memmap(f, dtype=np.uint16, mode="r")
                self.data_arrays.append(arr)
            self.total_tokens = sum(len(a) for a in self.data_arrays)
            self.cumulative = np.cumsum([len(a) for a in self.data_arrays])
            print(f"Loaded {len(self.files)} binary files, {self.total_tokens/1e6:.1f}M tokens")
        else:
            self.total_tokens = 0
            print(f"Found {len(self.jsonl_files)} jsonl files (not tokenized)")

    def get_batch(self, batch_size: int = 1) -> tuple:
        """Возвращает (x, y) батч."""
        if self.mode != "binary":
            raise NotImplementedError("Use binary mode")

        x = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        y = np.zeros((batch_size, self.seq_len), dtype=np.int64)

        for i in range(batch_size):
            # Random file
            file_idx = np.random.randint(len(self.data_arrays))
            arr = self.data_arrays[file_idx]
            # Random offset
            max_start = len(arr) - self.seq_len - 1
            if max_start <= 0:
                start = 0
            else:
                start = np.random.randint(0, max_start)
            chunk = arr[start:start + self.seq_len + 1]
            x[i] = chunk[:-1].astype(np.int64)
            y[i] = chunk[1:].astype(np.int64)

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_dataloader(self, batch_size: int = 1):
        """Бесконечный dataloader."""
        while True:
            yield self.get_batch(batch_size)
