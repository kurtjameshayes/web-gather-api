"""Regression tests for POST /ingest mode case-folding and null handling.

mode is lowercased immediately after JSON parse, before required-field checks.
JSON null therefore raises before the append/overwrite guard (generic HTTP 500).
Uppercase OVERWRITE is accepted as overwrite.
"""
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
    init_core(mock_mongo, mock_firecrawl, MagicMock())
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl}


@pytest.fixture
def client(mock_clients) -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    user_db = MagicMock(name="user_db")
    wg_db = MagicMock(name="wg_db")
    user_coll = MagicMock(name="user_coll")
    docs_coll = MagicMock(name="docs_coll")
    user_db.__getitem__.return_value = user_coll
    wg_db.__getitem__.return_value = docs_coll

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    user_coll.count_documents.return_value = 0
    return {"user_coll": user_coll, "docs_coll": docs_coll}


def test_ingest_uppercase_overwrite_is_accepted(client, mock_clients) -> None:
    """mode is case-folded, so OVERWRITE runs the overwrite path and echoes overwrite."""
    colls = _wire_mongo(mock_clients["mongo"])
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
                "mode": "OVERWRITE",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "overwrite"
    assert payload["overwritten"] is False
    assert payload["previous_document_count"] == 0
    colls["user_coll"].count_documents.assert_called_once_with(
        {"source_url": "https://example.com"}
    )
    colls["user_coll"].delete_many.assert_not_called()


def test_ingest_json_null_mode_is_generic_500_before_required_fields(
    client, mock_clients
) -> None:
    """JSON null mode currently calls .lower() before validation, which is a generic 500."""
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com",
            "database": "test_db",
            "collection": "docs",
            "mode": None,
        },
    )

    assert response.status_code == 500
    mock_clients["firecrawl"].crawl.assert_not_called()
