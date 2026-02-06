"""Endpoint tests for index and search flows."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
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


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_db = MagicMock()
    index_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        if name == "index_db":
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    source_db.__getitem__.return_value = MagicMock()
    index_db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
    return {"source_db": source_db, "index_db": index_db, "wg_db": wg_db}


def test_index_missing_required_params(client, mock_clients) -> None:
    response = client.post("/vector-index", json={})
    assert response.status_code == 400
    payload = response.get_json()
    assert "Missing required parameters" in payload["error"]

def test_index_embedding_model_not_configured(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    with patch("core.get_embedding_model_name", return_value=None):
        response = client.post(
            "/vector-index",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": "index_db",
                "index_collection_name": "chunks",
            },
        )
    assert response.status_code == 400
    payload = response.get_json()
    assert "embedding model" in payload["error"]


def test_index_invalid_source_query(client, mock_clients) -> None:
    _wire_mongo(mock_clients["mongo"])
    response = client.post(
        "/vector-index",
        json={
            "source_database_name": "src",
            "source_collection_name": "docs",
            "index_database_name": "index_db",
            "index_collection_name": "chunks",
            "source_query": "{not-json}",
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "source_query" in payload["error"]


def test_index_applies_source_query(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    source_collection = dbs["source_db"].__getitem__.return_value
    source_collection.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text"},
    ]

    class DummyModel:
        def encode(self, inputs, **kwargs: object) -> np.ndarray:
            return np.array([[1.0, 0.0] for _ in inputs])

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/vector-index",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": "index_db",
                "index_collection_name": "chunks",
                "source_query": '{"document_id": "doc-1"}',
            },
        )
    assert response.status_code == 200
    source_collection.find.assert_called_once_with({"document_id": "doc-1"})


def test_index_success(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["source_db"].__getitem__.return_value.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text"},
        {"_id": "row-2", "chunk_text": "More text"},
    ]

    class DummyModel:
        def encode(self, inputs, **kwargs: object) -> np.ndarray:
            return np.array([[1.0, 0.0] for _ in inputs])

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/vector-index",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": "index_db",
                "index_collection_name": "chunks",
            },
        )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunks_indexed"] == 2
    assert payload["embedding_model"] == "model"


def test_search_missing_params(client, mock_clients) -> None:
    response = client.get("/search")
    assert response.status_code == 400
    payload = response.get_json()
    assert "document_id" in payload["error"]


def test_search_document_not_found(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["wg_db"].__getitem__.return_value.find_one.return_value = None
    response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 404
    payload = response.get_json()
    assert "not found" in payload["error"]


def test_search_embedding_model_missing(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["wg_db"].__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "src",
        "collection_name": "docs",
    }
    with patch("core.get_embedding_model_name", return_value=None):
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 400
    payload = response.get_json()
    assert "embedding model" in payload["error"]


def test_search_no_chunks(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["wg_db"].__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "index_db",
        "collection_name": "docs",
        "chunk_collection": "chunks",
        "index_database_name": "index_db",
    }
    dbs["index_db"].__getitem__.return_value.find.return_value = []
    with patch("core.get_embedding_model_name", return_value="model"):
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 400
    payload = response.get_json()
    assert "no chunks" in payload["error"]


def test_search_success(client, mock_clients) -> None:
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["wg_db"].__getitem__.return_value.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "index_db",
        "collection_name": "docs",
        "chunk_collection": "chunks",
        "index_database_name": "index_db",
    }
    dbs["index_db"].__getitem__.return_value.find.return_value = [
        {"text": "match", "embedding": [1.0, 0.0]},
        {"text": "other", "embedding": [0.0, 1.0]},
    ]

    class DummyModel:
        def encode(self, inputs, **kwargs: object) -> np.ndarray:
            return np.array([[1.0, 0.0]])

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["results"][0]["text"] == "match"
