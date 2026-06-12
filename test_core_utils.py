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


@pytest.fixture
def clear_model_cache():
    core._model_cache.clear()
    yield
    core._model_cache.clear()


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
    clear_model_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    core._model_cache.update(
        {f"cached-model-{idx}": object() for idx in range(core._MAX_MODEL_CACHE_SIZE)}
    )
    model_factory = MagicMock()
    monkeypatch.setattr(core, "SentenceTransformer", model_factory)

    with pytest.raises(ValueError, match="cache full.*allowlist"):
        core.get_model("untrusted-model")

    model_factory.assert_not_called()


def test_get_model_evicts_oldest_entry_for_allowed_model(
    clear_model_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached_models = {
        f"cached-model-{idx}": object() for idx in range(core._MAX_MODEL_CACHE_SIZE)
    }
    core._model_cache.update(cached_models)
    loaded_model = object()
    model_factory = MagicMock(return_value=loaded_model)
    monkeypatch.setattr(core, "SentenceTransformer", model_factory)

    model = core.get_model("all-MiniLM-L6-v2")

    assert model is loaded_model
    model_factory.assert_called_once_with("all-MiniLM-L6-v2")
    assert "cached-model-0" not in core._model_cache
    assert core._model_cache["all-MiniLM-L6-v2"] is loaded_model
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE
