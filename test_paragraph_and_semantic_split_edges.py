"""Regression tests for paragraph overlap and semantic size splits.

Character, sentence, and semantic-similarity breaks are covered elsewhere.
Paragraph mode is a separate splitter: overlap is budgeted with the two
characters of '\\n\\n', a negative overlap does not drop paragraphs, and a
paragraph longer than chunk_size stays whole. Semantic mode copies the
sentence overlap loop but also force-splits on size even when similarity
never drops and the current chunk is still under the 20% floor.
"""
from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


_PARAS = "AAAA\n\nBBBB\n\nCCCC\n\nDDDD"


def test_paragraph_negative_overlap_keeps_every_paragraph() -> None:
    """Unlike character mode, a negative overlap only disables carry-forward."""
    chunks = core.chunk_text_by_paragraph(_PARAS, chunk_size=10, overlap=-3)

    assert chunks == ["AAAA\n\nBBBB", "CCCC\n\nDDDD"]
    assert chunks == core.chunk_text(_PARAS, chunk_size=10, overlap=-3, strategy="paragraph")


def test_paragraph_overlap_counts_the_blank_line_separator() -> None:
    """Two length-4 paragraphs are both carried only when overlap >= len + 2 + len.

    overlap 9 keeps one trailing paragraph. overlap 10 keeps two, because the
    second check is (4 + 2) + 4.
    """
    one_carried = core.chunk_text_by_paragraph(_PARAS, chunk_size=10, overlap=9)
    two_carried = core.chunk_text_by_paragraph(_PARAS, chunk_size=10, overlap=10)

    assert one_carried == [
        "AAAA\n\nBBBB",
        "BBBB\n\nCCCC",
        "CCCC\n\nDDDD",
    ]
    assert two_carried == [
        "AAAA\n\nBBBB",
        "AAAA\n\nBBBB\n\nCCCC",
        "BBBB\n\nCCCC\n\nDDDD",
    ]


def test_paragraph_longer_than_overlap_blocks_earlier_carry_and_stays_whole() -> None:
    """The overlap walk stops at the first paragraph that does not fit.

    A short paragraph behind a long one is not skipped forward into the next
    chunk, and the long paragraph is not split to satisfy chunk_size.
    """
    chunks = core.chunk_text_by_paragraph(
        "AAAA\n\nBBBBBBBBBB\n\nCC",
        chunk_size=12,
        overlap=6,
    )

    assert chunks == ["AAAA", "AAAA\n\nBBBBBBBBBB", "CC"]

    oversized = core.chunk_text_by_paragraph(("A" * 30) + "\n\nBB", chunk_size=10, overlap=0)
    assert oversized == ["A" * 30, "BB"]


def test_paragraph_single_newline_and_blank_input_do_not_split() -> None:
    """Only a blank line separates paragraphs. Empty input yields no chunks."""
    assert core.chunk_text_by_paragraph("Line one.\nLine two.", chunk_size=1000, overlap=0) == [
        "Line one.\nLine two."
    ]
    assert core.chunk_text_by_paragraph(None, chunk_size=10, overlap=0) == []
    assert core.chunk_text_by_paragraph("   \n\n  ", chunk_size=10, overlap=0) == []


class _FlatEmbeddings:
    """Adjacent sentences are identical, so similarity never drops below the threshold."""

    def encode(self, sentences, **kwargs):
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["normalize_embeddings"] is True
        count = len(sentences)
        return np.ones((count, 2)) / math.sqrt(2)


def test_semantic_size_break_ignores_the_twenty_percent_floor() -> None:
    """A sentence that would exceed chunk_size starts a new chunk even when
    the current chunk is still under 20% of chunk_size and similarity is high.
    """
    text = "Hi. " + ("A" * 100) + "."
    model = _FlatEmbeddings()

    split = core.chunk_text_by_semantic(text, chunk_size=100, overlap=0, model=model)
    kept = core.chunk_text_by_semantic(text, chunk_size=200, overlap=0, model=model)

    assert split == ["Hi.", "A" * 100 + "."]
    assert kept == ["Hi. " + ("A" * 100) + "."]


def test_semantic_size_overlap_matches_sentence_carry_forward() -> None:
    """When similarity does not split, semantic overlap uses the sentence rule."""
    text = "AAAA. BBBB. CCCC. DDDD."
    chunks = core.chunk_text_by_semantic(text, chunk_size=10, overlap=5, model=_FlatEmbeddings())

    assert chunks == core.chunk_text_by_sentence(text, chunk_size=10, overlap=5)
    assert chunks == ["AAAA.", "AAAA. BBBB.", "BBBB. CCCC.", "CCCC. DDDD."]


def test_semantic_whitespace_only_does_not_encode() -> None:
    model = MagicMock()

    assert core.chunk_text_by_semantic("   \n  ", chunk_size=50, overlap=0, model=model) == []
    model.encode.assert_not_called()
