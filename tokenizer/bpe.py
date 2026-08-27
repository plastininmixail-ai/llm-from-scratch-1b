"""
BPE-токенизатор с нуля на чистом Python (без сторонних библиотек).

Алгоритм Byte Pair Encoding:
  1. Базовый словарь = 256 байт UTF-8 (id 0..255 → bytes([0..255])).
  2. Каждый документ разбивается на "слова" по regex; слово = список токенов,
     каждый из которых пока что однобайтовый, плюс маркер конца слова </w> (id 256).
  3. Итеративно vocab_size - 257 раз:
       - считаем частоты всех соседних пар токенов;
       - самая частая пара → новый токен (склейка двух байтовых строк или
         более сложных токенов);
       - все вхождения этой пары в корпусе заменяются на новый токен.
  4. encode(text) — каждое слово представляется как список однобайтовых токенов
     + </w>; затем жадно по рангу merges склеиваем соседние токены.
  5. decode(ids) — собираем байты токенов, </w> даёт пробел между словами.

Особенности:
  - Полностью независим от сторонних библиотек (только stdlib).
  - Работает с UTF-8 (кириллица, эмодзи, CJK).
  - Сохранение в JSON: vocab (id → token) + merges (rank-ordered list).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------- #
# Regex для препроцессинга
# Python stdlib re не поддерживает \p{L}, поэтому используем [^\W\d_]
# (буквы любых алфавитов) и \d (цифры).
# ---------------------------------------------------------------------------- #
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|"""
    r"""[^\W\d_]+|\d+|"""
    r"""\s+\S+|\S+""",
    re.UNICODE,
)

# Маркер конца слова — отдельный id 256
END_OF_WORD = "</w>"


class BPETokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}        # id → bytes
        self.token_to_id: dict[bytes, int] = {}  # bytes → id
        self.merges: list[tuple[bytes, bytes]] = []  # rank-ordered merges
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        self.cache: dict[str, list[bytes]] = {}

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #
    def _init_vocab(self) -> None:
        """Базовый словарь: 256 байт + маркер </w>."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[256] = END_OF_WORD.encode("utf-8")
        self.token_to_id = {v: k for k, v in self.vocab.items()}

    def fit(self, text_iter: Iterable[str], vocab_size: int = 8000,
            min_pair_freq: int = 2, verbose: bool = True) -> "BPETokenizer":
        assert vocab_size >= 257, "vocab_size должен быть ≥ 257 (256 байт + </w>)"

        if verbose:
            print(f"[bpe] start: vocab_size={vocab_size}")

        self._init_vocab()

        if verbose:
            print("[bpe] reading corpus & tokenizing words...")

        # word_freqs: ключ = tuple bytes (токены), значение = частота слова
        word_freqs: Counter[tuple[bytes, ...]] = Counter()
        n_docs = 0
        for doc in text_iter:
            n_docs += 1
            for word in PAT.findall(doc):
                # слово → список байтов
                chars = tuple(bytes([b]) for b in word.encode("utf-8"))
                chars = chars + (END_OF_WORD.encode("utf-8"),)
                word_freqs[chars] += 1

        if verbose:
            print(f"[bpe] {n_docs} docs, {len(word_freqs)} unique words")

        # итеративно сливаем самые частые пары
        _t_start = time.time()
        iteration = 0
        while len(self.vocab) < vocab_size:
            pair_freqs: Counter[tuple[bytes, bytes]] = Counter()
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for a, b in zip(word[:-1], word[1:]):
                    pair_freqs[(a, b)] += freq

            if not pair_freqs:
                if verbose:
                    print("[bpe] no more pairs to merge")
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < min_pair_freq:
                if verbose:
                    print(f"[bpe] best pair freq {best_freq} < {min_pair_freq}, stop")
                break

            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_id] = new_token
            self.token_to_id[new_token] = new_id
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # пересчитываем корпус с новым токеном
            new_word_freqs: Counter[tuple[bytes, ...]] = Counter()
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if (i < len(word) - 1
                            and word[i] == best_pair[0]
                            and word[i+1] == best_pair[1]):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] += freq
            word_freqs = new_word_freqs

            iteration += 1
            if verbose and (iteration % 50 == 0 or iteration < 50):
                print(f"[bpe] vocab {len(self.vocab)}/{vocab_size} "
                      f"({iteration} merges done) "
                      f"last: {best_pair!r} → {len(new_token)} bytes "
                      f"(freq={best_freq}, time={time.time()-_t_start:.0f}s)",
                      flush=True)
            if iteration == 1:
                _t_start = time.time()

        if verbose:
            print(f"[bpe] done: vocab={len(self.vocab)}, merges={len(self.merges)}")

        self.cache = {}
        return self

    # ------------------------------------------------------------------ #
    # Кодирование
    # ------------------------------------------------------------------ #
    def _encode_word(self, word: str) -> list[bytes]:
        """Кодирует одно слово в список токенов (включая </w>)."""
        if word in self.cache:
            return self.cache[word]

        chars: list[bytes] = [bytes([b]) for b in word.encode("utf-8")]
        chars.append(END_OF_WORD.encode("utf-8"))

        # жадно склеиваем по рангу merges (меньший rank = раньше)
        # оптимизация: сначала все уникальные пары кэшируем, потом ищем минимум
        while len(chars) >= 2:
            best_idx = -1
            best_rank = float("inf")
            # локальный поиск с разом
            for i in range(len(chars) - 1):
                rank = self.merge_ranks.get((chars[i], chars[i + 1]),
                                              float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            chars = chars[:best_idx] + [chars[best_idx] + chars[best_idx + 1]] + chars[best_idx + 2:]

        self.cache[word] = chars
        return chars

    def encode(self, text: str) -> list[int]:
        """Кодирует текст в список id токенов."""
        ids: list[int] = []
        for word in PAT.findall(text):
            for tok in self._encode_word(word):
                tid = self.token_to_id.get(tok)
                if tid is not None:
                    ids.append(tid)
        return ids

    # ------------------------------------------------------------------ #
    # Декодирование
    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        """Декодирует список id обратно в текст."""
        byte_parts: list[bytes] = []
        eow_bytes = END_OF_WORD.encode("utf-8")
        for tid in ids:
            tok = self.vocab.get(tid)
            if tok is None:
                continue
            # ищем </w> внутри токена (после мерджей они могут быть склеены)
            # пример: токен = b'the</w>' → 'the' + ' ' после split
            if eow_bytes in tok:
                # разбиваем по </w>, последняя часть — после маркера
                pieces = tok.split(eow_bytes)
                # pieces: [before, between, ..., after_eow]
                # каждый </w> → пробел, последний кусок — без хвостового пробела
                for i, piece in enumerate(pieces):
                    if i > 0 and piece and (i < len(pieces) - 1 or True):
                        # добавляем пробел перед куском, если он не пустой
                        # и предыдущий кусок не заканчивался пробелом
                        if byte_parts and byte_parts[-1] not in (b" ", eow_bytes):
                            byte_parts.append(b" ")
                    if piece:
                        byte_parts.append(piece)
            else:
                byte_parts.append(tok)

        raw = b"".join(byte_parts)
        return raw.decode("utf-8", errors="replace").strip()

    # ------------------------------------------------------------------ #
    # Сохранение / загрузка
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [a.hex() + " " + b.hex() for a, b in self.merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        t = cls()
        t.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        t.token_to_id = {v: k for k, v in t.vocab.items()}
        t.merges = []
        for merge_str in data["merges"]:
            a_hex, b_hex = merge_str.split(" ", 1)
            t.merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        t.merge_ranks = {pair: i for i, pair in enumerate(t.merges)}
        return t

    # ------------------------------------------------------------------ #
    # Утилиты
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.vocab)

    def info(self) -> str:
        return (f"BPE tokenizer: vocab_size={len(self.vocab)}, "
                f"merges={len(self.merges)}")