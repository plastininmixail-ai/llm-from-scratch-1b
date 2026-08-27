"""
Тесты BPE-токенизатора.
Запуск: pytest tests/test_bpe.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer.bpe import BPETokenizer, END_OF_WORD, PAT


# ---------------------------------------------------------------------------- #
# Фикстуры
# ---------------------------------------------------------------------------- #
@pytest.fixture
def small_tokenizer():
    """Маленький BPE на крошечном корпусе."""
    tok = BPETokenizer()
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "hello world hello world",
        "программирование на python это интересно",
        "привет мир как дела",
        "def hello(): print(123)",
    ]
    tok.fit(corpus, vocab_size=400, verbose=False)
    return tok


# ---------------------------------------------------------------------------- #
# Базовые тесты
# ---------------------------------------------------------------------------- #
class TestTokenizerBasics:
    def test_vocab_has_byte_tokens(self, small_tokenizer):
        """vocab должен содержать все 256 байт + </w>."""
        for i in range(256):
            assert i in small_tokenizer.vocab
        assert 256 in small_tokenizer.vocab
        assert small_tokenizer.vocab[256] == END_OF_WORD.encode("utf-8")

    def test_vocab_size_at_least_257(self, small_tokenizer):
        # 256 байт + </w> = минимум 257, остальное — merges
        assert len(small_tokenizer) >= 257
        assert len(small_tokenizer) <= 400

    def test_merges_sorted_by_rank(self, small_tokenizer):
        """merges должны быть в порядке применения."""
        # merges и merge_ranks должны быть консистентны
        for i, (pair, rank) in enumerate(small_tokenizer.merge_ranks.items()):
            assert rank == i


# ---------------------------------------------------------------------------- #
# Round-trip тесты
# ---------------------------------------------------------------------------- #
class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "hello world",
        "привет мир",
        "the quick brown fox",
        "def hello(): print(123)",
        "Python is awesome",
        "Машинное обучение",
    ])
    def test_encode_decode_lossless(self, small_tokenizer, text):
        ids = small_tokenizer.encode(text)
        decoded = small_tokenizer.decode(ids)
        # кроме лидирующих/трейлинг пробелов, должно быть идентично
        assert decoded.strip() == text.strip(), (
            f"text={text!r} ids={ids} decoded={decoded!r}"
        )


# ---------------------------------------------------------------------------- #
# Свойства кодирования
# ---------------------------------------------------------------------------- #
class TestEncoding:
    def test_empty_string(self, small_tokenizer):
        assert small_tokenizer.encode("") == []

    def test_only_spaces_stripped(self, small_tokenizer):
        # крайние пробелы теряются (намеренно)
        ids = small_tokenizer.encode("   hello   ")
        assert small_tokenizer.decode(ids).strip() == "hello"

    def test_unknown_chars_fall_back(self, small_tokenizer):
        # emoji и спецсимволы: должны кодироваться как отдельные байты
        ids = small_tokenizer.encode("hello 🚀")
        assert len(ids) > 0
        # декодирование должно работать
        decoded = small_tokenizer.decode(ids)
        assert "hello" in decoded


# ---------------------------------------------------------------------------- #
# Сохранение / загрузка
# ---------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_round_trip_persistence(self, small_tokenizer, tmp_path):
        path = tmp_path / "vocab.json"
        small_tokenizer.save(path)
        loaded = BPETokenizer.load(path)

        assert len(loaded) == len(small_tokenizer)
        assert loaded.merges == small_tokenizer.merges

        # проверяем, что кодирование совпадает
        text = "the quick brown fox"
        assert loaded.encode(text) == small_tokenizer.encode(text)


# ---------------------------------------------------------------------------- #
# Регекс препроцессинга
# ---------------------------------------------------------------------------- #
class TestPreprocessing:
    def test_splits_words_and_punctuation(self):
        text = "Hello, world! How are you?"
        matches = PAT.findall(text)
        # регексп GPT-2-стиля захватывает " world!" как один токен с пробелом
        assert "Hello" in matches
        assert " world!" in matches  # " world!" (с пробелом) — один токен

    def test_handles_cyrillic(self):
        text = "Привет, мир!"
        matches = PAT.findall(text)
        assert "Привет" in matches
        assert " мир!" in matches  # кириллица тоже жадно ест пробелы

    def test_handles_contractions(self):
        text = "I'm don't can't"
        matches = PAT.findall(text)
        assert "I" in matches
        assert "'m" in matches

    def test_unicode_letters_recognized(self):
        # кириллические буквы должны распознаваться как часть слова
        text = "Привет"
        assert PAT.search(text) is not None