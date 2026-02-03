"""Endpoint tests for gather, ingest, and crawl flows."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from core import WEB_GATHER_DB, DOCUMENTS_COLLECTION, core_bp, init_core


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


def test_gather_success(client, mock_clients) -> None:
    mock_clients["firecrawl"].search.return_value = [
        {
            "url": "https://example.com/a",
            "title": "Privacy Policy",
            "description": "Data retention policy.",
        }
    ]
    response = client.post("/gather", json={"query": "privacy policy"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["query"] == "privacy policy"
    assert payload["results"][0]["score"] >= 0.0
    assert payload["results"][0]["percent_match"] >= 0.0
    assert payload["next_step"]["endpoint"] == "/ingest"


def test_gather_missing_query(client, mock_clients) -> None:
    response = client.post("/gather", json={})
    assert response.status_code == 400
    payload = response.get_json()
    assert "query" in payload["error"]


def test_ingest_web_success(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com", "title": "Title", "markdown": "Hello world"}
    ]
    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    assert payload["pages"] == 1


def test_ingest_pdf_success(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    with patch("core.is_pdf_url", return_value=True), patch(
        "core.download_pdf", return_value=b"%PDF-1.0"
    ), patch(
        "core.extract_text_from_pdf",
        return_value=[{"page_number": 1, "text": "PDF content"}],
    ), patch(
        "core.combine_pdf_pages", return_value="PDF content"
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/file.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "pdf"
    assert payload["pages"] == 1


def test_ingest_invalid_mode(client, mock_clients) -> None:
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com",
            "database": "test_db",
            "collection": "docs",
            "mode": "invalid",
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "mode" in payload["error"]


def test_crawl_puppeteer_success(client, mock_clients) -> None:
    with patch(
        "core.puppeteer_crawl",
        return_value=[{"url": "https://example.com", "title": "T", "markdown": "Body"}],
    ):
        response = client.post("/crawl", json={"url": "https://example.com"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "puppeteer"
    assert payload["pages_crawled"] == 1


def test_crawl_fallback_to_firecrawl(client, mock_clients) -> None:
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com", "title": "T", "markdown": "Body"}
    ]
    with patch("core.puppeteer_crawl", return_value=[]):
        response = client.post("/crawl", json={"url": "https://example.com"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "firecrawl"
    assert payload["pages_crawled"] == 1


def test_crawl_missing_url(client, mock_clients) -> None:
    response = client.post("/crawl", json={})
    assert response.status_code == 400
    payload = response.get_json()
    assert "url" in payload["error"]
