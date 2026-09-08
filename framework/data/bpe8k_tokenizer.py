"""BPE 8k tokenizer — адаптер для нового vocab."""
import json
from pathlib import Path
from typing import List

from framework.data.simple_tokenizer import SimpleBPETokenizer


class BPE8kTokenizer(SimpleBPETokenizer):
    """Простой byte-level encoder на основе vocab_8k.json (8000 tokens)."""

    def __init__(self, vocab_path: str = "tokenizer/vocab_8k.json"):
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.vocab = data.get("vocab", {})
        self.merges = data.get("merges", [])

        # byte_str -> id
        self.tok_to_id = {v: int(k) for k, v in self.vocab.items()}
        self.id_to_tok = {int(k): v for k, v in self.vocab.items()}

        # Build byte merges: (byte_a_str, byte_b_str) -> rank
        self.bpe_ranks = {}
        for i, m in enumerate(self.merges):
            if isinstance(m, str):
                # Может быть "65 3c2f773e" или просто токен
                parts = m.split(" ", 1)
                if len(parts) == 2:
                    a, b = parts
                    self.bpe_ranks[(a, b)] = i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
