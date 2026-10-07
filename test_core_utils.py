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


def test_chunk_text_by_semantic_uses_model() -> None:
    class DummyModel:
        def encode(self, sentences: List[str], **kwargs: object) -> np.ndarray:
            return np.eye(len(sentences))

    text = "First sentence. Second sentence. Third sentence."
    chunks = core.chunk_text_by_semantic(
        text, chunk_size=50, overlap=10, model=DummyModel()
    )
    assert chunks


@pytest.mark.parametrize(
    ("url", "allowed", "message_part"),
    [
        ("https://example.com/privacy", True, None),
        ("http://example.com:8080/a", True, None),
        ("file:///etc/passwd", False, "scheme"),
        ("ftp://example.com/file", False, "scheme"),
        ("http://", False, "hostname"),
        ("http://localhost/admin", False, "localhost"),
        ("http://LOCALHOST./admin", False, "localhost"),
        ("http://127.0.0.1:8080/", False, "private"),
        ("http://10.1.2.3/latest", False, "private"),
        ("http://192.168.1.10/", False, "private"),
        ("http://[::1]/admin", False, "localhost"),
        ("http://169.254.169.254/metadata", False, "private"),
        ("http://[fe80::1]/", False, "private"),
    ],
)
def test_validate_url_block_ssrf(url: str, allowed: bool, message_part: str | None) -> None:
    ok, message = core._validate_url_block_ssrf(url)
    assert ok is allowed
    if allowed:
        assert message is None
    else:
        assert message is not None
        assert message_part in message.lower()


@pytest.fixture
def clean_model_cache():
    saved = dict(core._model_cache)
    core._model_cache.clear()
    ctor = core.SentenceTransformer
    previous_return = ctor.return_value
    ctor.reset_mock()
    yield ctor
    core._model_cache.clear()
    core._model_cache.update(saved)
    ctor.reset_mock()
    ctor.return_value = previous_return


def test_get_model_cache_hit_skips_constructor(clean_model_cache: MagicMock) -> None:
    marker = object()
    clean_model_cache.return_value = marker
    first = core.get_model("all-MiniLM-L6-v2")
    second = core.get_model("all-MiniLM-L6-v2")
    assert first is marker
    assert second is marker
    assert clean_model_cache.call_count == 1


def test_get_model_rejects_unlisted_model_only_when_cache_is_full(
    clean_model_cache: MagicMock,
) -> None:
    marker = object()
    clean_model_cache.return_value = marker
    loaded = core.get_model("custom-unlisted-model")
    assert loaded is marker
    assert "custom-unlisted-model" in core._model_cache

    core._model_cache.clear()
    for index in range(core._MAX_MODEL_CACHE_SIZE):
        core._model_cache[f"cached-{index}"] = object()
    clean_model_cache.reset_mock()
    with pytest.raises(ValueError, match="not in the allowlist"):
        core.get_model("custom-unlisted-model")
    clean_model_cache.assert_not_called()
    assert "custom-unlisted-model" not in core._model_cache
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE


def test_get_model_evicts_oldest_allowlisted_model_when_full(
    clean_model_cache: MagicMock,
) -> None:
    for index in range(core._MAX_MODEL_CACHE_SIZE):
        core._model_cache[f"old-{index}"] = f"model-{index}"
    replacement = object()
    clean_model_cache.return_value = replacement

    loaded = core.get_model("all-MiniLM-L6-v2")
    assert loaded is replacement
    assert "old-0" not in core._model_cache
    assert core._model_cache["all-MiniLM-L6-v2"] is replacement
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE
    clean_model_cache.assert_called_once_with("all-MiniLM-L6-v2")


def test_combine_pages_prefers_markdown_and_skips_empty() -> None:
    class Metadata:
        url = "https://example.com/policy"
        title = "Privacy Policy"

    class Page:
        metadata = Metadata()
        markdown = "# We retain data."
        html = "<p>ignored html</p>"
        raw_html = "<html>ignored</html>"
        summary = "ignored summary"

    class EmptyPage:
        markdown = ""
        html = ""
        raw_html = None
        summary = ""

    combined = core.combine_pages(
        [
            EmptyPage(),
            Page(),
            {"title": "Contact", "url": "https://example.com/contact", "html": "<p>Email us</p>"},
        ]
    )
    assert "ignored html" not in combined
    assert "ignored summary" not in combined
    assert combined.index("Privacy Policy | Source: https://example.com/policy") < combined.index(
        "# We retain data."
    )
    assert "Contact | Source: https://example.com/contact" in combined
    assert "<p>Email us</p>" in combined
    assert combined.count("\n\n") == 1


def test_combine_pages_empty_input() -> None:
    assert core.combine_pages([]) == ""
