"""Unit tests for core utility helpers."""
from __future__ import annotations

import sys
from typing import Any, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

import core


@pytest.fixture
def isolated_model_cache() -> None:
    original_cache = dict(core._model_cache)
    core._model_cache.clear()
    try:
        yield
    finally:
        core._model_cache.clear()
        core._model_cache.update(original_cache)


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


def test_chunk_text_by_semantic_uses_model() -> None:
    class DummyModel:
        def encode(self, sentences: List[str], **kwargs: object) -> np.ndarray:
            return np.eye(len(sentences))

    text = "First sentence. Second sentence. Third sentence."
    chunks = core.chunk_text_by_semantic(
        text, chunk_size=50, overlap=10, model=DummyModel()
    )
    assert chunks


def test_get_model_rejects_unknown_model_when_cache_full(
    monkeypatch: Any, isolated_model_cache: None
) -> None:
    for idx in range(core._MAX_MODEL_CACHE_SIZE):
        core._model_cache[f"cached-{idx}"] = object()
    loader = MagicMock()
    monkeypatch.setattr(core, "SentenceTransformer", loader)

    with pytest.raises(ValueError, match="not in the allowlist"):
        core.get_model("untrusted-model-name")

    loader.assert_not_called()
    assert list(core._model_cache) == [
        f"cached-{idx}" for idx in range(core._MAX_MODEL_CACHE_SIZE)
    ]


def test_get_model_evicts_oldest_allowed_model_when_cache_full(
    monkeypatch: Any, isolated_model_cache: None
) -> None:
    for idx in range(core._MAX_MODEL_CACHE_SIZE):
        core._model_cache[f"cached-{idx}"] = object()
    loaded_model = object()
    loader = MagicMock(return_value=loaded_model)
    monkeypatch.setattr(core, "SentenceTransformer", loader)

    result = core.get_model("all-MiniLM-L6-v2")

    assert result is loaded_model
    loader.assert_called_once_with("all-MiniLM-L6-v2")
    assert "cached-0" not in core._model_cache
    assert core._model_cache["all-MiniLM-L6-v2"] is loaded_model
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE
