"""Unit tests for core utility helpers."""
from __future__ import annotations

import sys
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

import core


def test_calculate_relevance_score_exact_phrase() -> None:
    score = core.calculate_relevance_score(
        query="data retention",
        title="Data Retention Policy",
        description="Retention for data is required.",
    )
    assert score == 1.0


def test_calculate_relevance_score_partial_match() -> None:
    score = core.calculate_relevance_score(
        query="data retention",
        title="Data policy",
        description="Retention guidance.",
    )
    assert score == 0.5


def test_calculate_relevance_score_empty_and_description_boost() -> None:
    """Empty queries score 0. Phrase boost is +0.1 in the description and caps at 1."""
    assert core.calculate_relevance_score("", "Data", "Retention") == 0.0
    assert core.calculate_relevance_score(None, "Data", "Retention") == 0.0
    assert core.calculate_relevance_score("...", "Data", "Retention") == 0.0
    assert core.calculate_relevance_score("data", None, None) == 0.0

    # Both terms hit the description only: 2/6, then +0.1 because the phrase is there.
    description_only = core.calculate_relevance_score(
        query="data retention",
        title="Other topic",
        description="See data retention rules",
    )
    assert description_only == 0.433

    # Term match is substring containment, so "cat" hits "category" and takes the +0.2 title boost.
    substring_hit = core.calculate_relevance_score(
        query="cat",
        title="Category guide",
        description="",
    )
    assert substring_hit == 0.867

    capped = core.calculate_relevance_score(
        query="data",
        title="data extra",
        description="data",
    )
    assert capped == 1.0


def test_serialize_search_result_scores_and_metadata() -> None:
    """Gather results replace upstream scores and prefer metadata url/title when set."""
    raw = {
        "title": "Data Retention",
        "description": "Retention for data is required.",
        "score": 9,
    }
    scored = core.serialize_search_result(raw, query="data retention")
    assert raw["score"] == 9
    assert scored["score"] == 1.0
    assert scored["percent_match"] == 100.0

    unscored = core.serialize_search_result({"title": "Data"}, query="")
    assert "score" not in unscored
    assert "percent_match" not in core.serialize_search_result({"title": "Data"}, query=None)

    partial = core.serialize_search_result(
        {"title": "alpha", "description": ""},
        query="alpha beta gamma",
    )
    assert partial["score"] == 0.222
    assert partial["percent_match"] == 22.2

    class _Meta:
        def __init__(self, url: str | None, title: str | None) -> None:
            self.url = url
            self.title = title

    class _Page:
        def __init__(self, metadata: _Meta) -> None:
            self.url = "https://direct.example/a"
            self.title = "Direct"
            self.description = "Direct description"
            self.markdown = "body"
            self.metadata = metadata

    fallback = core.serialize_search_result(_Page(_Meta("", None)), query=None)
    assert fallback["url"] == "https://direct.example/a"
    assert fallback["title"] == "Direct"
    assert fallback["markdown"] == "body"
    assert "score" not in fallback

    overridden = core.serialize_search_result(
        _Page(_Meta("https://meta.example/b", "Meta Title")),
        query=None,
    )
    assert overridden["url"] == "https://meta.example/b"
    assert overridden["title"] == "Meta Title"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/file.pdf", True),
        ("https://example.com/download?file=pdf", True),
        ("https://example.com/page.txt", False),
    ],
)
def test_is_pdf_url(url: str, expected: bool) -> None:
    assert core.is_pdf_url(url) is expected


def test_chunk_text_semantic_falls_back_to_sentence() -> None:
    text = "First sentence. Second sentence. Third sentence."
    expected = core.chunk_text_by_sentence(text, chunk_size=50, overlap=0)
    actual = core.chunk_text(text, chunk_size=50, overlap=0, strategy="semantic")
    assert actual == expected


def test_chunk_text_by_semantic_uses_model() -> None:
    class DummyModel:
        def encode(self, sentences: List[str], **kwargs: object) -> np.ndarray:
            return np.eye(len(sentences))

    text = "First sentence. Second sentence. Third sentence."
    chunks = core.chunk_text_by_semantic(
        text, chunk_size=50, overlap=10, model=DummyModel()
    )
    assert chunks
