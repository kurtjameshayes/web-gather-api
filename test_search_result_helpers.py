"""Leftover gather/search helper contracts not covered by existing unit tests.

PR coverage already pins Firecrawl payload normalization and scored object
serialization. These cases pin metadata URL/title override, score omission
when no query is passed, empty/punctuation ranking, and case-insensitive PDF
detection used by ingest.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


def test_serialize_search_result_metadata_overrides_direct_fields() -> None:
    """Firecrawl Document metadata.url/title replace object-level fields when set."""
    result = SimpleNamespace(
        url="https://example.com/direct",
        title="Direct Title",
        description="Rules for data retention.",
        markdown="# Body",
        metadata=SimpleNamespace(
            url="https://example.com/from-metadata",
            title="Meta Title",
        ),
    )
    serialized = core.serialize_search_result(result, query="data retention")
    assert serialized["url"] == "https://example.com/from-metadata"
    assert serialized["title"] == "Meta Title"
    assert serialized["score"] > 0
    assert serialized["percent_match"] == round(serialized["score"] * 100, 1)


def test_serialize_search_result_without_query_omits_score_and_copies_dict() -> None:
    """No query means no ranking keys; dict inputs must not be mutated in place."""
    original = {"url": "https://example.com/x", "title": "Other", "description": "n/a"}
    serialized = core.serialize_search_result(original, query=None)
    assert serialized == original
    assert "score" not in serialized
    assert "percent_match" not in serialized
    serialized_with_query = core.serialize_search_result(original, query="other")
    assert "score" in serialized_with_query
    assert "score" not in original


def test_calculate_relevance_score_empty_or_punctuation_query_is_zero() -> None:
    assert core.calculate_relevance_score("", "Title", "Description") == 0.0
    assert core.calculate_relevance_score("   ", "Title", "Description") == 0.0
    assert core.calculate_relevance_score("???", "Title", "Description") == 0.0


def test_calculate_relevance_score_boosts_exact_phrase_in_description() -> None:
    """Exact phrase in description (not title) currently adds a 0.1 boost."""
    without_phrase = core.calculate_relevance_score(
        "data retention",
        "Privacy Policy",
        "Retention of data is documented.",
    )
    with_phrase = core.calculate_relevance_score(
        "data retention",
        "Privacy Policy",
        "Our data retention schedule is annual.",
    )
    assert with_phrase == round(min(without_phrase + 0.1, 1.0), 3)
    assert with_phrase > without_phrase


def test_is_pdf_url_is_case_insensitive_on_path() -> None:
    assert core.is_pdf_url("https://example.com/FILE.PDF") is True
    assert core.is_pdf_url("https://example.com/docs/Report.Pdf?download=1") is True
    assert core.is_pdf_url("https://example.com/FILE.TXT") is False
