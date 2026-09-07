"""Simple encoder/decoder на основе vocab_3k.json (byte-level)."""
import json
from pathlib import Path
from typing import List


class SimpleBPETokenizer:
    """Простой byte-level encoder на основе vocab файла."""

    def __init__(self, vocab_path: str = "tokenizer/vocab_3k.json"):
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.vocab = data.get("vocab", {})
        self.merges = data.get("merges", [])

        # byte_str -> id
        self.tok_to_id = {v: int(k) for k, v in self.vocab.items()}

        # id -> byte_str
        self.id_to_tok = {int(k): v for k, v in self.vocab.items()}

        # Build byte merges: (byte_a, byte_b) -> merged_byte
        self.bpe_ranks = {}
        for i, m in enumerate(self.merges):
            if isinstance(m, (list, tuple)) and len(m) == 2:
                a, b = m
                if isinstance(a, str) and isinstance(b, str):
                    self.bpe_ranks[(a, b)] = i

    def encode(self, text: str) -> List[int]:
        """Encode text в ids (с BPE если возможно)."""
        if not text:
            return [1]

        # Применить BPE
        words = text.split(" ")
        ids = []
        for i, word in enumerate(words):
            if i > 0:
                ids.append(self.tok_to_id.get(" ", 1))
            word_ids = self._encode_word(word)
            ids.extend(word_ids)
        return ids if ids else [1]

    def _encode_word(self, word: str) -> List[int]:
        """Encode одно слово."""
        if not word:
            return []

        # Начальное представление: каждый символ как byte hex
        symbols = [f"{ord(c):02x}" for c in word]

        # Применить merges
        while len(symbols) > 1:
            # Найти пару с минимальным rank
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
            if best_idx == -1:
                break
            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1]
            symbols.pop(best_idx + 1)

        # Преобразовать в ids
        return [self.tok_to_id.get(s, 1) for s in symbols]

    def decode(self, ids: List[int]) -> str:
        """Decode ids в text."""
        out = []
        for i in ids:
            byte_str = self.id_to_tok.get(int(i), "")
            if byte_str and len(byte_str) % 2 == 0:
                try:
                    chars = []
                    for j in range(0, len(byte_str), 2):
                        byte = int(byte_str[j:j+2], 16)
                        chars.append(chr(byte))
                    out.append("".join(chars))
                except Exception:
                    pass
        return "".join(out)
