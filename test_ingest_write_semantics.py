"""Regression tests for /ingest write payloads and empty-content guards.

Base tests only assert status/document_type. Overwrite-scope deletes are covered
in open PRs; the insert record shape and "do not write empty documents" paths
were still untested on v0_1.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
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


def _wire_ingest_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_coll = MagicMock()
    meta_coll = MagicMock()
    source_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else source_db

    def _get_source_coll(name: str) -> MagicMock:
        return source_coll

    def _get_wg_coll(name: str) -> MagicMock:
        return meta_coll

    mock_mongo.__getitem__.side_effect = _get_db
    source_db.__getitem__.side_effect = _get_source_coll
    wg_db.__getitem__.side_effect = _get_wg_coll
    return {"source_coll": source_coll, "meta_coll": meta_coll}


def test_ingest_web_writes_document_and_metadata(client, mock_clients) -> None:
    """Successful web ingest must persist source_url/text and a documents catalog row."""
    colls = _wire_ingest_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/privacy", "title": "Privacy", "markdown": "We honor deletion requests."}
    ]

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/privacy",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    assert payload["pages"] == 1
    assert payload["mode"] == "append"
    assert payload["document_id"]

    colls["source_coll"].insert_one.assert_called_once()
    stored = colls["source_coll"].insert_one.call_args[0][0]
    assert stored["_id"] == payload["document_id"]
    assert stored["source_url"] == "https://example.com/privacy"
    assert stored["document_type"] == "web"
    assert "We honor deletion requests." in stored["text"]
    assert "created_at" in stored

    colls["meta_coll"].insert_one.assert_called_once()
    meta = colls["meta_coll"].insert_one.call_args[0][0]
    assert meta["document_id"] == payload["document_id"]
    assert meta["database_name"] == "test_db"
    assert meta["collection_name"] == "docs"
    assert meta["source_url"] == "https://example.com/privacy"
    assert meta["document_type"] == "web"
    assert "We honor deletion requests." in meta["description"]
    colls["source_coll"].delete_many.assert_not_called()


def test_ingest_zero_pages_does_not_write(client, mock_clients) -> None:
    """A crawl that returns no pages must 400 without inserting documents."""
    colls = _wire_ingest_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = []

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/blocked",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 400
    assert "no pages" in response.get_json()["error"].lower()
    colls["source_coll"].insert_one.assert_not_called()
    colls["meta_coll"].insert_one.assert_not_called()


def test_ingest_empty_combined_text_does_not_write(client, mock_clients) -> None:
    """Pages with no markdown/text must 400 without writing empty documents."""
    colls = _wire_ingest_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/empty", "title": "Empty", "markdown": ""}
    ]

    with patch("core.detect_content_type", return_value=""), patch(
        "core.combine_pages", return_value=""
    ):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/empty",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 400
    assert "no content" in response.get_json()["error"].lower()
    colls["source_coll"].insert_one.assert_not_called()
    colls["meta_coll"].insert_one.assert_not_called()


def test_ingest_pdf_without_text_does_not_write(client, mock_clients) -> None:
    """PDFs with no extractable text must 400 without inserting documents."""
    colls = _wire_ingest_mongo(mock_clients["mongo"])

    with patch("core.is_pdf_url", return_value=True), patch(
        "core.download_pdf", return_value=b"%PDF-1.0"
    ), patch("core.extract_text_from_pdf", return_value=[]):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/blank.pdf",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 400
    assert "no extractable text" in response.get_json()["error"].lower()
    colls["source_coll"].insert_one.assert_not_called()
    colls["meta_coll"].insert_one.assert_not_called()


def test_ingest_crawl_exception_does_not_write(client, mock_clients) -> None:
    """Firecrawl failures must 500 without partial writes."""
    colls = _wire_ingest_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.side_effect = RuntimeError("site unreachable")

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/down",
                "database": "test_db",
                "collection": "docs",
            },
        )

    assert response.status_code == 500
    assert "crawl failed" in response.get_json()["error"].lower()
    colls["source_coll"].insert_one.assert_not_called()
    colls["meta_coll"].insert_one.assert_not_called()
