"""Regression tests for /ingest PDF vs web routing from Content-Type."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import DOCUMENTS_COLLECTION, WEB_GATHER_DB, core_bp, init_core


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


def test_ingest_uses_pdf_path_when_head_content_type_is_pdf(client, mock_clients) -> None:
    """URLs without a .pdf suffix must still take the PDF path when HEAD says PDF."""
    _wire_mongo(mock_clients["mongo"])
    url = "https://example.com/download?id=99"
    with patch("core.detect_content_type", return_value="application/pdf; charset=binary") as detect, patch(
        "core.download_pdf", return_value=b"%PDF-1.4"
    ) as download, patch(
        "core.extract_text_from_pdf",
        return_value=[{"page_number": 1, "text": "Policy PDF text"}],
    ) as extract, patch(
        "core.combine_pdf_pages", return_value="Policy PDF text"
    ):
        response = client.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "pdf"
    assert payload["pages"] == 1
    detect.assert_called_once_with(url)
    download.assert_called_once_with(url)
    extract.assert_called_once()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_uses_web_crawl_when_head_content_type_is_html(client, mock_clients) -> None:
    """Non-PDF Content-Type must keep the Firecrawl path and skip PDF download."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/policy", "title": "Policy", "markdown": "Hello"}
    ]
    with patch("core.detect_content_type", return_value="text/html; charset=utf-8") as detect, patch(
        "core.download_pdf"
    ) as download:
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/policy",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    detect.assert_called_once_with("https://example.com/policy")
    download.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_called_once()
    # Catalog row is still written for web ingest.
    wg_docs = mock_clients["mongo"][WEB_GATHER_DB][DOCUMENTS_COLLECTION]
    assert wg_docs.insert_one.called
