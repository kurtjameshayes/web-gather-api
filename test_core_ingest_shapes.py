"""Firecrawl result shaping, PDF detection, and PDF text assembly.

These helpers decide whether gather/ingest keeps pages or routes a URL
through the PDF parser.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

import core


def test_normalize_results_prefers_data_list_over_web() -> None:
    data = [{"url": "https://example.com/a"}]
    web = [{"url": "https://example.com/b"}]
    result = core.normalize_results(SimpleNamespace(data=data, web=web))
    assert result is data


def test_normalize_results_uses_web_when_data_is_not_a_list() -> None:
    web = [{"url": "https://example.com/search"}]
    result = core.normalize_results(SimpleNamespace(data=None, web=web))
    assert result is web


def test_normalize_results_ignores_none_web_and_unknown_objects() -> None:
    assert core.normalize_results(SimpleNamespace(data=None, web=None)) == []
    assert core.normalize_results(SimpleNamespace()) == []
    assert core.normalize_results(None) == []
    assert core.normalize_results("plain") == []


def test_normalize_results_dict_precedence_and_unrecognized_web_key() -> None:
    data = [{"id": 1}]
    results = [{"id": 2}]
    pages = [{"id": 3}]
    assert core.normalize_results({"data": data, "results": results, "pages": pages}) is data
    assert core.normalize_results({"results": results, "pages": pages}) is results
    assert core.normalize_results({"pages": pages}) is pages
    # A null "data" value is logged with len() before it is returned, so gather
    # and ingest raise instead of falling through to "results".
    with pytest.raises(TypeError):
        core.normalize_results({"data": None, "results": results})
    assert core.normalize_results({"results": []}) == []
    # Dict search payloads use "web"; only object attributes honor that name.
    assert core.normalize_results({"web": [{"url": "https://example.com"}]}) == []


def test_normalize_results_returns_a_bare_list() -> None:
    rows = [{"url": "https://example.com"}]
    assert core.normalize_results(rows) is rows
    assert core.normalize_results([]) == []


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://cdn.example.com/FILE.PDF", True),
        ("https://example.com/doc?format=PDF", True),
        ("https://example.com/page?topic=notpdf", True),
        ("https://example.com/files/pdf/readme", False),
        ("https://example.com/file.pdf.bak", False),
        ("https://example.com/page.txt#pdf", False),
    ],
)
def test_is_pdf_url_case_and_substring(url: str, expected: bool) -> None:
    assert core.is_pdf_url(url) is expected


def test_combine_pdf_pages_skips_empty_text_and_defaults_page_number() -> None:
    combined = core.combine_pdf_pages(
        [
            {"page_number": 1, "text": ""},
            {"text": "  still kept  "},
            {"page_number": 3, "text": "Body"},
        ],
        "https://example.com/policy.pdf",
    )
    assert combined == (
        "Page ? | Source: https://example.com/policy.pdf\n  still kept  "
        "\n\n"
        "Page 3 | Source: https://example.com/policy.pdf\nBody"
    )
    assert core.combine_pdf_pages([], "https://example.com/policy.pdf") == ""


def test_extract_text_from_pdf_strips_and_skips_blank_pages() -> None:
    page_text = MagicMock()
    page_text.extract_text.return_value = "  Hello  "
    page_blank = MagicMock()
    page_blank.extract_text.return_value = "   "
    page_none = MagicMock()
    page_none.extract_text.return_value = None
    reader = MagicMock()
    reader.pages = [page_text, page_blank, page_none]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, "PdfReader", MagicMock(return_value=reader))
        pages = core.extract_text_from_pdf(b"%PDF-1.4")
    assert pages == [{"page_number": 1, "text": "Hello"}]


def test_extract_text_from_pdf_reraises_reader_errors() -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, "PdfReader", MagicMock(side_effect=RuntimeError("bad pdf")))
        with pytest.raises(RuntimeError, match="bad pdf"):
            core.extract_text_from_pdf(b"not-a-pdf")
