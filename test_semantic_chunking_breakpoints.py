"""Semantic chunking breakpoints and strategy dispatcher.

Base coverage only checks that a dummy model returns some chunks. These tests
pin unknown-strategy fallback, empty/single-sentence short-circuits, similarity
breakpoints, size-constraint splits, and overlap carry-forward.
"""
from __future__ import annotations

import sys
from typing import List
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


class ScriptedModel:
    """Returns a pre-assigned embedding per sentence, in call order."""

    def __init__(self, vectors: List[List[float]]) -> None:
        self._vectors = np.array(vectors, dtype=float)
        self.seen: List[List[str]] = []

    def encode(self, sentences: List[str], **kwargs: object) -> np.ndarray:
        self.seen.append(list(sentences))
        assert kwargs.get("normalize_embeddings") is True
        assert len(sentences) == len(self._vectors)
        return self._vectors


def test_chunk_text_unknown_strategy_falls_back_to_character() -> None:
    text = "abcdefghij"
    expected = core.chunk_text_by_character(text, chunk_size=4, overlap=0)
    actual = core.chunk_text(text, chunk_size=4, overlap=0, strategy="not-a-strategy")
    assert actual == expected
    assert actual  # dispatcher must not return empty for non-empty input


def test_chunk_text_by_semantic_empty_and_single_sentence() -> None:
    model = ScriptedModel([[1.0, 0.0]])
    assert core.chunk_text_by_semantic("", chunk_size=50, overlap=0, model=model) == []
    assert core.chunk_text_by_semantic("   ", chunk_size=50, overlap=0, model=model) == []
    # Single sentence short-circuits before encode().
    assert core.chunk_text_by_semantic(
        "Only one sentence.", chunk_size=50, overlap=0, model=model
    ) == ["Only one sentence."]
    assert model.seen == []


def test_chunk_text_by_semantic_breaks_on_similarity_drop() -> None:
    """Adjacent low-similarity sentences start a new chunk once the current one is large enough."""
    s1 = "Alpha privacy collection rules apply here."
    s2 = "Alpha privacy collection rules continue now."
    s3 = "Zeta cookies tracking advertising practices."
    s4 = "Zeta cookies tracking advertising continue."
    text = f"{s1} {s2} {s3} {s4}"
    model = ScriptedModel(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    # chunk_size large enough to avoid a size split, small enough that two
    # sentences exceed the 20% minimum required before a semantic break.
    chunks = core.chunk_text_by_semantic(text, chunk_size=200, overlap=0, model=model)
    assert chunks == [f"{s1} {s2}", f"{s3} {s4}"]


def test_chunk_text_by_semantic_size_constraint_splits_similar_sentences() -> None:
    s1 = "First similar sentence stays together."
    s2 = "Second similar sentence exceeds limit."
    text = f"{s1} {s2}"
    model = ScriptedModel([[1.0, 0.0], [1.0, 0.0]])
    chunks = core.chunk_text_by_semantic(text, chunk_size=len(s1), overlap=0, model=model)
    assert chunks == [s1, s2]


def test_chunk_text_by_semantic_overlap_keeps_trailing_sentences() -> None:
    s1 = "Topic one sentence number one."
    s2 = "Topic one sentence number two."
    s3 = "Topic two sentence number one."
    text = f"{s1} {s2} {s3}"
    model = ScriptedModel(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    chunks = core.chunk_text_by_semantic(
        text, chunk_size=200, overlap=len(s2) + 5, model=model
    )
    assert len(chunks) == 2
    assert chunks[0] == f"{s1} {s2}"
    assert chunks[1] == f"{s2} {s3}"
