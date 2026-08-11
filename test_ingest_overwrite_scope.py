"""Regression tests for /ingest overwrite scoping (source_url-only deletes)."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import WEB_GATHER_DB, core_bp, init_core


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


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_overwrite_collections(mock_clients):
    db = MagicMock()
    wg_db = MagicMock()
    target_collection = MagicMock()
    documents_collection = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_clients["mongo"].__getitem__.side_effect = _get_db
    db.__getitem__.return_value = target_collection
    wg_db.__getitem__.return_value = documents_collection
    return target_collection, documents_collection


def test_ingest_overwrite_deletes_only_matching_source_url(client, mock_clients) -> None:
    """Overwrite must clear only docs for the same source_url, never the whole collection."""
    url = "https://example.com/privacy"
    target_collection, documents_collection = _wire_overwrite_collections(mock_clients)
    target_collection.count_documents.return_value = 2
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": url, "title": "Privacy", "markdown": "Policy body"}
    ]

    with patch("core.detect_content_type", return_value=""), patch(
        "core.uuid.uuid4", return_value="doc-1"
    ):
        response = client.post(
            "/ingest",
            json={
                "url": url,
                "database": "test_db",
                "collection": "docs",
                "mode": "overwrite",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "overwrite"
    assert payload["overwritten"] is True
    assert payload["previous_document_count"] == 2

    overwrite_filter = {"source_url": url}
    target_collection.count_documents.assert_called_once_with(overwrite_filter)
    target_collection.delete_many.assert_called_once_with(overwrite_filter)
    documents_collection.delete_many.assert_called_once_with(
        {
            "database_name": "test_db",
            "collection_name": "docs",
            "source_url": url,
        }
    )
    # Guard against regressing to delete_many({}) which wipes unrelated documents.
    assert target_collection.delete_many.call_args.args[0] != {}

    inserted_doc = target_collection.insert_one.call_args.args[0]
    assert inserted_doc["_id"] == "doc-1"
    assert inserted_doc["source_url"] == url


def test_ingest_overwrite_skips_delete_when_no_prior_docs(client, mock_clients) -> None:
    """When no prior docs match the source URL, overwrite inserts without deletes."""
    url = "https://example.com/new-page"
    target_collection, documents_collection = _wire_overwrite_collections(mock_clients)
    target_collection.count_documents.return_value = 0
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": url, "title": "New", "markdown": "Fresh content"}
    ]

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": url,
                "database": "test_db",
                "collection": "docs",
                "mode": "overwrite",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["overwritten"] is False
    assert payload["previous_document_count"] == 0
    target_collection.delete_many.assert_not_called()
    documents_collection.delete_many.assert_not_called()
    target_collection.insert_one.assert_called_once()


def test_ingest_append_does_not_delete(client, mock_clients) -> None:
    """Append mode must never call delete_many on target or documents collections."""
    url = "https://example.com/append-me"
    target_collection, documents_collection = _wire_overwrite_collections(mock_clients)
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": url, "title": "Append", "markdown": "More content"}
    ]

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": url,
                "database": "test_db",
                "collection": "docs",
                "mode": "append",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "append"
    assert "overwritten" not in payload
    target_collection.delete_many.assert_not_called()
    documents_collection.delete_many.assert_not_called()
    target_collection.insert_one.assert_called_once()
    assert documents_collection.insert_one.call_args.args[0]["collection_name"] == "docs"
