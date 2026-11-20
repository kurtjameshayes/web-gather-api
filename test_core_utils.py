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


@pytest.fixture
def clean_model_cache():
    saved = dict(core._model_cache)
    core._model_cache.clear()
    try:
        yield
    finally:
        core._model_cache.clear()
        core._model_cache.update(saved)


def test_get_model_allows_unknown_name_until_cache_is_full(clean_model_cache, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: List[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name
            constructed.append(name)

    monkeypatch.setattr(core, "SentenceTransformer", FakeModel)
    model = core.get_model("org/custom-embedder")
    assert model.name == "org/custom-embedder"
    assert core._model_cache["org/custom-embedder"] is model
    assert constructed == ["org/custom-embedder"]

    again = core.get_model("org/custom-embedder")
    assert again is model
    assert constructed == ["org/custom-embedder"]


def test_get_model_rejects_unknown_name_when_cache_is_full(clean_model_cache, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: List[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            constructed.append(name)

    monkeypatch.setattr(core, "SentenceTransformer", FakeModel)
    for index in range(core._MAX_MODEL_CACHE_SIZE):
        core.get_model(f"custom-{index}")
    cached = dict(core._model_cache)

    with pytest.raises(ValueError, match="not-a-real-model.*not in the allowlist"):
        core.get_model("not-a-real-model")

    assert "not-a-real-model" not in constructed
    assert core._model_cache == cached


def test_get_model_evicts_oldest_insertion_not_least_recently_used(
    clean_model_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(core, "SentenceTransformer", FakeModel)
    for index in range(core._MAX_MODEL_CACHE_SIZE):
        core.get_model(f"custom-{index}")
    core.get_model("custom-0")

    allowed = "all-MiniLM-L6-v2"
    loaded = core.get_model(allowed)

    assert "custom-0" not in core._model_cache
    assert "custom-1" in core._model_cache
    assert core._model_cache[allowed] is loaded
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE


def test_chunk_text_by_semantic_uses_model() -> None:
    class DummyModel:
        def encode(self, sentences: List[str], **kwargs: object) -> np.ndarray:
            return np.eye(len(sentences))

    text = "First sentence. Second sentence. Third sentence."
    chunks = core.chunk_text_by_semantic(
        text, chunk_size=50, overlap=10, model=DummyModel()
    )
    assert chunks
