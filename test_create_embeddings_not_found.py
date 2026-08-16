"""Regression tests for /create-embeddings empty-source and query-type guards.

Write-semantic coverage in open PRs asserts insert/update payloads. These
tests pin the 404 empty-source path and non-object source_query rejection
that still only have status-only or no coverage on base.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import core_bp, init_core


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


def _wire_source(mock_mongo: MagicMock) -> MagicMock:
    source_db = MagicMock()
    index_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == "web-gather":
            return wg_db
        if name == "index_db":
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    source_coll = MagicMock()
    source_db.__getitem__.return_value = source_coll
    index_db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
    return source_coll


def _index_body(**overrides: object) -> dict:
    body = {
        "source_database_name": "src",
        "source_collection_name": "docs",
        "index_database_name": "index_db",
        "index_collection_name": "chunks",
    }
    body.update(overrides)
    return body


def test_create_embeddings_no_source_documents_returns_404(client, mock_clients) -> None:
    source_coll = _wire_source(mock_clients["mongo"])
    source_coll.find.return_value = []

    with patch("core.get_embedding_model_name", return_value="model"):
        response = client.post("/create-embeddings", json=_index_body())

    assert response.status_code == 404
    payload = response.get_json()
    assert "No documents found" in payload["error"]
    assert "src.docs" in payload["error"]
    source_coll.find.assert_called_once_with({})


def test_create_embeddings_rejects_list_source_query(client, mock_clients) -> None:
    response = client.post(
        "/create-embeddings",
        json=_index_body(source_query=["not", "an", "object"]),
    )
    assert response.status_code == 400
    assert "source_query must be a JSON object" in response.get_json()["error"]


def test_create_embeddings_rejects_json_array_source_query(client, mock_clients) -> None:
    response = client.post(
        "/create-embeddings",
        json=_index_body(source_query="[{\"document_id\":\"doc-1\"}]"),
    )
    assert response.status_code == 400
    assert "source_query must be a JSON object" in response.get_json()["error"]
