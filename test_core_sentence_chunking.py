"""Regression tests for sentence-boundary chunking used by semantic fallback."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


def test_chunk_text_by_sentence_empty_input() -> None:
    assert core.chunk_text_by_sentence("") == []
    assert core.chunk_text_by_sentence("   ") == []


def test_chunk_text_by_sentence_respects_size_and_overlap() -> None:
    text = "Alpha sentence. Bravo sentence. Charlie sentence. Delta sentence."
    chunks = core.chunk_text_by_sentence(text, chunk_size=30, overlap=20)

    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)
    # Overlap should keep a trailing sentence from the previous chunk when it fits.
    assert any(
        chunks[i].split()[-1] in chunks[i + 1]
        for i in range(len(chunks) - 1)
    )


def test_chunk_text_dispatcher_sentence_strategy() -> None:
    text = "One. Two. Three."
    assert core.chunk_text(text, chunk_size=20, overlap=5, strategy="sentence") == (
        core.chunk_text_by_sentence(text, chunk_size=20, overlap=5)
    )
