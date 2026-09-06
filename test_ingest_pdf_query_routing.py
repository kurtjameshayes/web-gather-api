"""Ingest routes to PDF vs web from URL query strings without mocking is_pdf_url."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import WEB_GATHER_DB, core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


def _wire_mongo(mock_mongo: MagicMock) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()


def _pdf_patches():
    return (
        patch("core.download_pdf", return_value=b"%PDF-1.0"),
        patch(
            "core.extract_text_from_pdf",
            return_value=[{"page_number": 1, "text": "PDF content"}],
        ),
        patch("core.combine_pdf_pages", return_value="PDF content"),
        patch("core.detect_content_type"),
    )


def test_ingest_query_file_pdf_skips_head(client, mock_clients) -> None:
    """A ?file=pdf query is treated as PDF and never issues a HEAD content-type check."""
    _wire_mongo(mock_clients["mongo"])
    download, extract, combine, detect = _pdf_patches()
    with download as download_mock, extract, combine, detect as detect_mock:
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/doc?file=pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    assert response.get_json()["document_type"] == "pdf"
    download_mock.assert_called_once()
    detect_mock.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_query_format_pdf_is_case_insensitive(client, mock_clients) -> None:
    """Query values are lowercased, so ?format=PDF also takes the PDF download path."""
    _wire_mongo(mock_clients["mongo"])
    download, extract, combine, detect = _pdf_patches()
    with download, extract, combine, detect as detect_mock:
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/doc?format=PDF",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    assert response.get_json()["document_type"] == "pdf"
    detect_mock.assert_not_called()


def test_ingest_query_substring_notpdf_is_treated_as_pdf(client, mock_clients) -> None:
    """Current contract: any query containing the substring 'pdf' is a PDF URL."""
    _wire_mongo(mock_clients["mongo"])
    download, extract, combine, detect = _pdf_patches()
    with download, extract, combine, detect as detect_mock:
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/page?notpdf=1",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    assert response.get_json()["document_type"] == "pdf"
    detect_mock.assert_not_called()


def test_ingest_html_url_uses_head_then_firecrawl(client, mock_clients) -> None:
    """URLs without a PDF path or query still HEAD-check, then crawl as web."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/page", "title": "Title", "markdown": "Hello"}
    ]
    with patch("core.detect_content_type", return_value="text/html") as detect_mock:
        with patch("core.download_pdf") as download_mock:
            response = client.post(
                "/ingest",
                json={
                    "url": "https://example.com/page",
                    "database": "test_db",
                    "collection": "docs",
                },
            )
    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    detect_mock.assert_called_once_with("https://example.com/page")
    download_mock.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_called_once()
