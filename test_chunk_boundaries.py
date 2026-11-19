"""Chunk overlap and boundary contracts used by indexing."""
from __future__ import annotations

import numpy as np

import core


def test_character_chunks_overlap_and_drop_whitespace_only_slices() -> None:
    assert core.chunk_text_by_character("abcdefghij", chunk_size=4, overlap=2) == [
        "abcd",
        "cdef",
        "efgh",
        "ghij",
    ]
    assert core.chunk_text_by_character("abcdefghij", chunk_size=4, overlap=0) == [
        "abcd",
        "efgh",
        "ij",
    ]
    # overlap == chunk_size - 1 still advances by one character.
    assert core.chunk_text_by_character("abcde", chunk_size=3, overlap=2) == [
        "abc",
        "bcd",
        "cde",
    ]
    assert core.chunk_text_by_character("  ab  cd  ", chunk_size=4, overlap=0) == ["ab", "cd"]
    assert core.chunk_text_by_character("", chunk_size=4, overlap=1) == []
    assert core.chunk_text_by_character(None, chunk_size=4, overlap=1) == []
    assert core.chunk_text_by_character("hello", chunk_size=100, overlap=1000) == ["hello"]


def test_sentence_chunks_keep_overlap_with_separator_accounting() -> None:
    text = "AAAAA. BBBBB. CCCCC."
    # Each sentence is 6 characters. Overlap 12 keeps only the last sentence:
    # the separator's extra character makes 6 + 1 + 6 exceed 12.
    assert core.chunk_text_by_sentence(text, chunk_size=13, overlap=12) == [
        "AAAAA. BBBBB.",
        "BBBBB. CCCCC.",
    ]
    assert core.chunk_text_by_sentence(text, chunk_size=6, overlap=0) == [
        "AAAAA.",
        "BBBBB.",
        "CCCCC.",
    ]
    # A sentence longer than chunk_size is not split.
    assert core.chunk_text_by_sentence("ABCDEFGHIJ. Next.", chunk_size=5, overlap=0) == [
        "ABCDEFGHIJ.",
        "Next.",
    ]
    assert core.chunk_text_by_sentence("no terminator here", chunk_size=5, overlap=0) == [
        "no terminator here",
    ]
    assert core.chunk_text_by_sentence("", chunk_size=10, overlap=0) == []
    assert core.chunk_text_by_sentence("   ", chunk_size=10, overlap=0) == []


def test_paragraph_chunks_split_on_blank_lines_and_count_separators() -> None:
    assert core.chunk_text_by_paragraph("Hello\nworld\n\nNext", chunk_size=1000, overlap=0) == [
        "Hello\nworld\n\nNext",
    ]
    # Five-character paragraphs. Overlap 11 keeps only the last one because
    # the blank-line separator adds 2 to the running overlap length.
    assert core.chunk_text_by_paragraph(
        "AAAAA\n\nBBBBB\n\nCCCCC",
        chunk_size=12,
        overlap=11,
    ) == [
        "AAAAA\n\nBBBBB",
        "BBBBB\n\nCCCCC",
    ]
    assert core.chunk_text_by_paragraph("ABCDEFGHIJ\n\nXY", chunk_size=4, overlap=0) == [
        "ABCDEFGHIJ",
        "XY",
    ]
    assert core.chunk_text_by_paragraph("\n\n\n", chunk_size=10, overlap=0) == []


def test_unknown_strategy_falls_back_to_character() -> None:
    text = "abcdefghij"
    assert core.chunk_text(text, chunk_size=4, overlap=0, strategy="nope") == [
        "abcd",
        "efgh",
        "ij",
    ]


class _Vectors:
    def encode(self, sentences, **kwargs):
        table = {
            "Aaa.": [1.0, 0.0],
            "Bbb.": [1.0, 0.0],
            "Ccc.": [0.0, 1.0],
            "Ddd.": [0.0, 1.0],
        }
        return np.array([table[sentence] for sentence in sentences], dtype=float)


def test_semantic_break_respects_minimum_chunk_fraction() -> None:
    text = "Aaa. Bbb. Ccc. Ddd."
    # Similarity drops between Bbb and Ccc. chunk_size 40 is small enough that
    # the first pair exceeds 20% of the budget, so the drop starts a new chunk.
    assert core.chunk_text_by_semantic(text, chunk_size=40, overlap=0, model=_Vectors()) == [
        "Aaa. Bbb.",
        "Ccc. Ddd.",
    ]
    # The same drop is ignored when the current chunk is still under 20%.
    assert core.chunk_text_by_semantic(text, chunk_size=1000, overlap=0, model=_Vectors()) == [
        "Aaa. Bbb. Ccc. Ddd.",
    ]


def test_semantic_high_similarity_breaks_only_on_size() -> None:
    class _Same:
        def encode(self, sentences, **kwargs):
            return np.array([[1.0, 0.0]] * len(sentences), dtype=float)

    assert core.chunk_text_by_semantic(
        "Aaa. Bbb. Ccc.",
        chunk_size=10,
        overlap=0,
        model=_Same(),
    ) == [
        "Aaa. Bbb.",
        "Ccc.",
    ]
