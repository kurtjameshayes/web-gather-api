"""Regression tests for POST /ingest overwrite when no matching source_url docs exist.

Overwrite must still insert the new document and report overwritten=false without
calling delete_many when previous_document_count is 0. Distinct from scoped
deletes that run only when matching documents already exist.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

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


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_db = MagicMock()
    wg_db = MagicMock()
    source_coll = MagicMock()
    wg_coll = MagicMock()
    source_db.__getitem__.return_value = source_coll
    wg_db.__getitem__.return_value = wg_coll

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else source_db

    mock_mongo.__getitem__.side_effect = _get_db
    return {"source_coll": source_coll, "wg_coll": wg_coll, "wg_db": wg_db}


def test_ingest_overwrite_zero_existing_docs_skips_delete(
    client, mock_clients
) -> None:
    """Overwrite with no matching source_url docs must not wipe; still inserts."""
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["source_coll"].count_documents.return_value = 0
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/page", "title": "Title", "markdown": "Hello world"}
    ]

    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/page",
                "database": "test_db",
                "collection": "docs",
                "mode": "overwrite",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "overwrite"
    assert payload["overwritten"] is False
    assert payload["previous_document_count"] == 0
    assert payload["document_type"] == "web"
    assert "document_id" in payload

    dbs["source_coll"].count_documents.assert_called_once_with(
        {"source_url": "https://example.com/page"}
    )
    dbs["source_coll"].delete_many.assert_not_called()
    dbs["wg_coll"].delete_many.assert_not_called()

    dbs["source_coll"].insert_one.assert_called_once()
    inserted = dbs["source_coll"].insert_one.call_args[0][0]
    assert inserted["source_url"] == "https://example.com/page"
    assert inserted["document_type"] == "web"
    assert "Hello world" in inserted["text"]

    dbs["wg_coll"].insert_one.assert_called_once()
    meta = dbs["wg_coll"].insert_one.call_args[0][0]
    assert meta["document_id"] == payload["document_id"]
    assert meta["database_name"] == "test_db"
    assert meta["collection_name"] == "docs"
    assert meta["source_url"] == "https://example.com/page"
    dbs["wg_db"].__getitem__.assert_called_with(DOCUMENTS_COLLECTION)
