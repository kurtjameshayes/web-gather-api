"""Regression tests for the text crawl and PDF ingest store.

Crawl results prefer a list on .data, then .web. Page text prefers markdown,
then html, raw html, and summary. Empty metadata hides a page's own URL.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())

import core


def test_normalize_results_prefers_data_list_over_web() -> None:
    raw = SimpleNamespace(
        data=[{"markdown": "from-data"}],
        web=[{"markdown": "from-web"}],
    )

    assert core.normalize_results(raw) == [{"markdown": "from-data"}]


def test_normalize_results_uses_web_when_data_is_not_a_list() -> None:
    raw = SimpleNamespace(data="not-a-list", web=[{"markdown": "from-web"}])

    assert core.normalize_results(raw) == [{"markdown": "from-web"}]


def test_normalize_results_returns_an_empty_web_list() -> None:
    assert core.normalize_results(SimpleNamespace(web=[])) == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"data": [{"markdown": "data"}], "results": [{"markdown": "results"}], "pages": [{"markdown": "pages"}]},
            [{"markdown": "data"}],
        ),
        (
            {"results": [{"markdown": "results"}], "pages": [{"markdown": "pages"}]},
            [{"markdown": "results"}],
        ),
        ({"pages": [{"markdown": "pages"}]}, [{"markdown": "pages"}]),
    ],
)
def test_normalize_results_dict_key_order(payload: dict, expected: list) -> None:
    assert core.normalize_results(payload) == expected


def test_normalize_results_returns_a_list_unchanged_and_rejects_other_types() -> None:
    pages = [{"markdown": "kept"}]

    assert core.normalize_results(pages) is pages
    assert core.normalize_results("nope") == []
    assert core.normalize_results(None) == []
    assert core.normalize_results(SimpleNamespace(data=None, web=None)) == []


def test_combine_pages_uses_markdown_before_later_bodies_and_skips_empty_pages() -> None:
    pages = [
        {
            "url": "https://example.com/policy",
            "title": "Privacy",
            "markdown": "Policy body",
            "html": "<p>ignored</p>",
            "summary": "ignored summary",
        },
        {
            "url": "https://example.com/empty",
            "title": "Empty",
            "markdown": "",
            "html": "",
            "raw_html": "",
            "summary": "",
        },
        {
            "url": "https://example.com/html",
            "html": "<p>Keep</p>",
            "raw_html": "raw ignored",
            "summary": "summary ignored",
        },
    ]

    assert core.combine_pages(pages) == (
        "Privacy | Source: https://example.com/policy\nPolicy body"
        "\n\n"
        "Source: https://example.com/html\n<p>Keep</p>"
    )


def test_whitespace_markdown_blocks_the_html_fallback() -> None:
    page = {
        "url": "https://example.com/policy",
        "title": "Privacy",
        "markdown": "  \n  ",
        "html": "<p>real body</p>",
    }

    assert core.combine_pages([page]) == "Privacy | Source: https://example.com/policy"


def test_raw_html_is_used_before_summary() -> None:
    page = {
        "url": "https://example.com/raw",
        "markdown": None,
        "html": "",
        "raw_html": "RAW",
        "summary": "SUM",
    }

    assert core.combine_pages([page]) == "Source: https://example.com/raw\nRAW"


def test_dict_metadata_is_ignored() -> None:
    page = {
        "metadata": {"url": "https://nested.example", "title": "Nested"},
        "url": "https://top.example",
        "title": "Top",
        "markdown": "Body",
    }

    assert core.combine_pages([page]) == "Top | Source: https://top.example\nBody"


def test_empty_object_metadata_hides_the_direct_url_and_title() -> None:
    page = SimpleNamespace(
        metadata=SimpleNamespace(url="", title=""),
        url="https://direct.example/page",
        title="Direct Title",
        markdown="Direct body",
        html="<p>ignored</p>",
    )

    assert core.combine_pages([page]) == "Source: \nDirect body"


def test_object_metadata_supplies_the_stored_source_line() -> None:
    page = SimpleNamespace(
        metadata=SimpleNamespace(url="https://meta.example/p", title="Meta Title"),
        url="https://direct.example/p",
        title="Direct Title",
        markdown="Body text",
        html="<p>ignored</p>",
    )

    assert core.combine_pages([page]) == (
        "Meta Title | Source: https://meta.example/p\nBody text"
    )


def test_normalize_then_combine_uses_the_data_list() -> None:
    raw = SimpleNamespace(
        data=[
            {"title": "A", "url": "https://a.example", "markdown": "Alpha"},
            {"title": "B", "url": "https://b.example", "html": "Beta"},
        ],
        web=[{"markdown": "should-not-appear"}],
    )

    assert core.combine_pages(core.normalize_results(raw)) == (
        "A | Source: https://a.example\nAlpha"
        "\n\n"
        "B | Source: https://b.example\nBeta"
    )


def test_combine_pages_empty_input_is_empty() -> None:
    assert core.combine_pages([]) == ""


def test_combine_pdf_pages_skips_blank_text_and_numbers_missing_pages() -> None:
    pages = [
        {"page_number": 2, "text": "  Second  "},
        {"text": ""},
        {"page_number": 1, "text": "First"},
        {},
        {"text": "Only"},
    ]

    assert core.combine_pdf_pages(pages, "https://example.com/a.pdf") == (
        "Page 2 | Source: https://example.com/a.pdf\n  Second  "
        "\n\n"
        "Page 1 | Source: https://example.com/a.pdf\nFirst"
        "\n\n"
        "Page ? | Source: https://example.com/a.pdf\nOnly"
    )


def test_combine_pdf_pages_empty_input_is_empty() -> None:
    assert core.combine_pdf_pages([], "https://example.com/a.pdf") == ""
    assert core.combine_pdf_pages([{"page_number": 1, "text": ""}], "https://example.com/a.pdf") == ""
