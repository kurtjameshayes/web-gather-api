"""Regression tests for SSRF URL validation and crawl/search helper utilities.

These helpers gate ingest/crawl network access and normalize Firecrawl payloads
before persistence/search scoring. Failures here have a large blast radius.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

import core


@pytest.mark.parametrize(
    ("url", "ok", "error_substr"),
    [
        ("https://example.com/policy", True, None),
        ("http://docs.example.org/a", True, None),
        ("ftp://example.com/file", False, "scheme"),
        ("file:///etc/passwd", False, "scheme"),
        ("https://localhost/admin", False, "localhost"),
        ("http://127.0.0.1:8080/", False, "private"),
        ("http://10.0.0.5/internal", False, "private"),
        ("http://192.168.1.10/x", False, "private"),
        ("http://169.254.169.254/latest/meta-data", False, "private"),
        ("http://[::1]/", False, "localhost"),
        ("http://[fe80::1]/", False, "private"),
        ("not-a-url", False, "scheme"),
        ("https://", False, "hostname"),
    ],
)
def test_validate_url_block_ssrf_blocks_unsafe_targets(
    url: str, ok: bool, error_substr: str | None
) -> None:
    """Ingest/crawl must reject non-http(s), localhost, and private/link-local IPs."""
    allowed, err = core._validate_url_block_ssrf(url)
    assert allowed is ok
    if ok:
        assert err is None
    else:
        assert err is not None
        assert error_substr in err.lower()


def test_validate_url_block_ssrf_allows_public_hostname() -> None:
    """Public hostnames remain allowed; DNS resolution is not blocked here."""
    allowed, err = core._validate_url_block_ssrf("https://policies.acme.com/privacy")
    assert allowed is True
    assert err is None


def test_normalize_results_supports_firecrawl_shapes() -> None:
    """Crawl/search normalization must accept object and dict Firecrawl payloads."""
    crawl_job = SimpleNamespace(data=[{"url": "https://a.example"}])
    search_data = SimpleNamespace(web=[{"url": "https://b.example"}])

    assert core.normalize_results(crawl_job) == [{"url": "https://a.example"}]
    assert core.normalize_results(search_data) == [{"url": "https://b.example"}]
    assert core.normalize_results({"data": [1]}) == [1]
    assert core.normalize_results({"results": [2]}) == [2]
    assert core.normalize_results({"pages": [3]}) == [3]
    assert core.normalize_results([{"url": "https://c.example"}]) == [
        {"url": "https://c.example"}
    ]
    assert core.normalize_results("unexpected") == []
    assert core.normalize_results(None) == []


def test_get_page_attr_reads_object_and_dict() -> None:
    page_obj = SimpleNamespace(markdown="from-attr", html="")
    page_dict = {"markdown": "", "html": "from-dict"}

    assert core.get_page_attr(page_obj, "markdown") == "from-attr"
    assert core.get_page_attr(page_obj, "html", default="fallback") == "fallback"
    assert core.get_page_attr(page_dict, "html") == "from-dict"
    assert core.get_page_attr(page_dict, "missing", default="x") == "x"
    assert core.get_page_attr(42, "markdown", default="none") == "none"


def test_combine_pages_prefers_markdown_and_skips_empty() -> None:
    """Combined crawl text must prefer markdown and ignore pages without content."""
    pages = [
        {
            "url": "https://example.com/a",
            "title": "Alpha",
            "markdown": "Policy text A",
            "html": "<p>ignored</p>",
        },
        {
            "url": "https://example.com/b",
            "title": "",
            "html": "Fallback HTML B",
        },
        {"url": "https://example.com/empty"},
        SimpleNamespace(
            metadata=SimpleNamespace(url="https://example.com/c", title="Charlie"),
            markdown="Policy text C",
            html="",
            raw_html="",
            summary="",
        ),
    ]

    combined = core.combine_pages(pages)
    assert "Alpha | Source: https://example.com/a" in combined
    assert "Policy text A" in combined
    assert "Source: https://example.com/b" in combined
    assert "Fallback HTML B" in combined
    assert "Charlie | Source: https://example.com/c" in combined
    assert "Policy text C" in combined
    assert "example.com/empty" not in combined
    assert "<p>ignored</p>" not in combined


def test_serialize_search_result_scores_query_matches() -> None:
    """Gather serialization must attach relevance score/percent_match for ranking."""
    result = SimpleNamespace(
        url="https://example.com/retention",
        title="Data Retention Policy",
        description="Rules for keeping data.",
        markdown="# Retention",
        metadata=None,
    )
    serialized = core.serialize_search_result(result, query="data retention")
    assert serialized["url"] == "https://example.com/retention"
    assert serialized["title"] == "Data Retention Policy"
    assert serialized["score"] == 1.0
    assert serialized["percent_match"] == 100.0

    as_dict = core.serialize_search_result(
        {"url": "https://example.com/x", "title": "Other", "description": ""},
        query="data retention",
    )
    assert as_dict["score"] == 0.0
    assert as_dict["percent_match"] == 0.0


def test_cosine_similarity_ranks_aligned_vectors() -> None:
    """Vector search scoring must return empty for no chunks and rank aligned vectors higher."""
    query = np.array([1.0, 0.0])
    assert core.cosine_similarity(query, []).size == 0

    chunks = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.0])]
    scores = core.cosine_similarity(query, chunks)
    assert scores.tolist() == [1.0, 0.0, 0.5]
    assert scores.argmax() == 0


def test_chunk_text_by_character_honors_overlap_and_empty() -> None:
    """Character chunking must produce overlapping windows and treat empty text as no chunks."""
    assert core.chunk_text_by_character("", chunk_size=10, overlap=2) == []
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = core.chunk_text_by_character(text, chunk_size=10, overlap=3)
    assert chunks == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
