"""PDF page extraction used by POST /ingest.

Blank and missing page text must not become stored documents, and a reader
failure must surface as a parse error rather than an empty success.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import (
    WEB_GATHER_DB,
    combine_pdf_pages,
    core_bp,
    extract_text_from_pdf,
    init_core,
)


def _page(text: object) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def test_extract_text_drops_blank_pages_and_keeps_original_numbers() -> None:
    reader = MagicMock()
    reader.pages = [
        _page("  \n"),
        _page(None),
        _page("  Consumer notice.  "),
    ]
    with patch("core.PdfReader", return_value=reader) as ctor:
        pages = extract_text_from_pdf(b"%PDF-1.4 fake")

    stream = ctor.call_args.args[0]
    assert stream.read() == b"%PDF-1.4 fake"
    assert pages == [{"page_number": 3, "text": "Consumer notice."}]


def test_extract_text_reraises_reader_errors() -> None:
    with patch("core.PdfReader", side_effect=ValueError("corrupt")):
        with pytest.raises(ValueError, match="corrupt"):
            extract_text_from_pdf(b"not-a-pdf")


def test_combine_pdf_pages_skips_empty_text_and_numbers_missing_pages() -> None:
    combined = combine_pdf_pages(
        [
            {"page_number": 1, "text": ""},
            {"page_number": 0, "text": "Zero is a real page"},
            {"text": "Missing number"},
            {"page_number": 4, "text": "Last"},
        ],
        "https://example.com/policy.pdf",
    )
    assert combined == (
        "Page 0 | Source: https://example.com/policy.pdf\nZero is a real page"
        "\n\nPage ? | Source: https://example.com/policy.pdf\nMissing number"
        "\n\nPage 4 | Source: https://example.com/policy.pdf\nLast"
    )


def test_combine_pdf_pages_empty_input_is_blank() -> None:
    assert combine_pdf_pages([], "https://example.com/policy.pdf") == ""
    assert combine_pdf_pages([{"page_number": 1, "text": ""}], "https://example.com/a.pdf") == ""


def test_combine_pdf_pages_whitespace_text_keeps_header_only() -> None:
    # The page text is truthy, so a header is emitted. The final strip then
    # removes the whitespace-only body.
    assert combine_pdf_pages(
        [{"page_number": 1, "text": "   "}],
        "https://example.com/a.pdf",
    ) == "Page 1 | Source: https://example.com/a.pdf"


@pytest.fixture
def app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.register_blueprint(core_bp)
    return flask_app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl}


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    db = MagicMock()
    wg_db = MagicMock()
    collection = MagicMock()
    documents = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = collection
    wg_db.__getitem__.return_value = documents
    return {"collection": collection, "documents": documents}


def test_ingest_blank_pdf_is_rejected_before_insert(client, mock_clients) -> None:
    wired = _wire_mongo(mock_clients["mongo"])
    reader = MagicMock()
    reader.pages = [_page("   "), _page(None)]
    with patch("core.download_pdf", return_value=b"%PDF"), patch(
        "core.PdfReader", return_value=reader
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/empty.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "PDF has no extractable text"
    wired["collection"].insert_one.assert_not_called()
    wired["documents"].insert_one.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_corrupt_pdf_is_parse_failure_before_insert(client, mock_clients) -> None:
    wired = _wire_mongo(mock_clients["mongo"])
    with patch("core.download_pdf", return_value=b"%PDF"), patch(
        "core.PdfReader", side_effect=ValueError("corrupt")
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://cdn.example.com/policy.PDF",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert response.get_json()["error"] == (
        "Failed to parse PDF. The file may be corrupted or unsupported."
    )
    wired["collection"].insert_one.assert_not_called()
    wired["documents"].insert_one.assert_not_called()
