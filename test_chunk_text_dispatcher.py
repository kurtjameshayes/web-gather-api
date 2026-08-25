"""Regression tests for core.chunk_text strategy dispatch and unknown-strategy fallback."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules["sentence_transformers"] = MagicMock()

import core


def test_unknown_strategy_falls_back_to_character() -> None:
    """Unrecognized strategies must use character splitting, not raise."""
    text = "abcdefghijklmnopqrstuvwxyz" * 8
    expected = core.chunk_text_by_character(text, chunk_size=40, overlap=8)
    actual = core.chunk_text(
        text, chunk_size=40, overlap=8, strategy="not-a-real-strategy"
    )
    assert actual == expected
    assert actual


def test_sentence_strategy_keeps_sentence_boundaries() -> None:
    """Sentence strategy must not split mid-sentence the way character splitting does."""
    text = "First sentence. Second sentence. Third sentence."
    by_char = core.chunk_text(text, chunk_size=20, overlap=0, strategy="character")
    by_sent = core.chunk_text(text, chunk_size=20, overlap=0, strategy="sentence")
    assert by_sent != by_char
    assert by_sent[0] == "First sentence."
    assert all(chunk.endswith(".") for chunk in by_sent)


def test_paragraph_strategy_splits_on_blank_lines() -> None:
    """Paragraph strategy splits on double newlines rather than character windows."""
    text = "Para one is here.\n\nPara two is here.\n\nPara three is here."
    chunks = core.chunk_text(text, chunk_size=20, overlap=0, strategy="paragraph")
    assert len(chunks) >= 2
    assert "Para one is here." in chunks[0]
    assert "Para two is here." in chunks[1]


def test_character_overlap_advances_without_duplicating_whole_window() -> None:
    """Overlap smaller than chunk_size must advance the window (no stall)."""
    text = "abcdefghij" * 6  # 60 chars
    chunks = core.chunk_text_by_character(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 3
    assert chunks[0] == text[:20]
    # Next window starts at end - overlap = 15
    assert chunks[1] == text[15:35]
