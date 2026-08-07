"""Regression tests for PDF download/extraction helpers used by ingest."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Avoid loading heavy optional deps during import.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


def test_download_pdf_returns_response_bytes() -> None:
    mock_response = MagicMock()
    mock_response.content = b"%PDF-fake"
    mock_response.raise_for_status = MagicMock()

    with patch("core.requests.get", return_value=mock_response) as mock_get:
        content = core.download_pdf("https://example.com/doc.pdf")

    assert content == b"%PDF-fake"
    mock_get.assert_called_once_with("https://example.com/doc.pdf", timeout=60)
    mock_response.raise_for_status.assert_called_once()


def test_download_pdf_propagates_http_errors() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = RuntimeError("http failed")

    with patch("core.requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="http failed"):
            core.download_pdf("https://example.com/missing.pdf")


def test_extract_text_from_pdf_skips_blank_pages() -> None:
    page_with_text = MagicMock()
    page_with_text.extract_text.return_value = "Keep me"
    blank_page = MagicMock()
    blank_page.extract_text.return_value = "   "

    reader = MagicMock()
    reader.pages = [blank_page, page_with_text]

    with patch("core.PdfReader", return_value=reader):
        pages = core.extract_text_from_pdf(b"%PDF-mock")

    assert pages == [{"page_number": 2, "text": "Keep me"}]


def test_extract_text_from_pdf_raises_on_reader_failure() -> None:
    with patch("core.PdfReader", side_effect=ValueError("corrupt pdf")):
        with pytest.raises(ValueError, match="corrupt pdf"):
            core.extract_text_from_pdf(b"not-a-pdf")


def test_combine_pdf_pages_joins_headers_and_skips_empty() -> None:
    combined = core.combine_pdf_pages(
        [
            {"page_number": 1, "text": "Alpha"},
            {"page_number": 2, "text": ""},
            {"page_number": 3, "text": "Gamma"},
        ],
        "https://example.com/a.pdf",
    )
    assert combined.startswith("Page 1 | Source: https://example.com/a.pdf\nAlpha")
    assert "Page 2" not in combined
    assert "Page 3 | Source: https://example.com/a.pdf\nGamma" in combined
    assert "\n\n" in combined


def test_detect_content_type_returns_header_value() -> None:
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "Application/PDF; charset=binary"}

    with patch("core.requests.head", return_value=mock_response) as mock_head:
        content_type = core.detect_content_type("https://example.com/file")

    assert content_type == "application/pdf; charset=binary"
    mock_head.assert_called_once_with(
        "https://example.com/file",
        allow_redirects=True,
        timeout=10,
    )


def test_detect_content_type_returns_empty_string_on_failure() -> None:
    with patch("core.requests.head", side_effect=RuntimeError("timeout")):
        assert core.detect_content_type("https://example.com/file") == ""
