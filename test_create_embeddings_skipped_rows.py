"""Regression tests for POST /create-embeddings skipped-row accounting.

Blank/whitespace source rows must be skipped without blocking indexing of
remaining rows, and skipped_rows must be reported.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

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


class _DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in inputs])


def test_create_embeddings_counts_skipped_blank_rows(client, mock_clients) -> None:
    """None / empty / whitespace rows are skipped; remaining rows are indexed."""
    dbs = _wire_mongo(mock_clients["mongo"])
    source_collection = dbs["source_db"].__getitem__.return_value
    index_collection = dbs["index_db"].__getitem__.return_value
    source_collection.find.return_value = [
        {"_id": "row-1", "chunk_text": "Keep this"},
        {"_id": "row-2", "chunk_text": None},
        {"_id": "row-3", "chunk_text": ""},
        {"_id": "row-4", "chunk_text": "   "},
        {"_id": "row-5", "chunk_text": "Also keep"},
    ]
    index_collection.delete_many.return_value = MagicMock(deleted_count=0)

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=_DummyModel()
    ):
        response = client.post(
            "/create-embeddings",
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
    assert payload["skipped_rows"] == 3
    inserted = index_collection.insert_many.call_args[0][0]
    assert len(inserted) == 2
    assert [doc["source_id"] for doc in inserted] == ["row-1", "row-5"]
    assert [doc["chunk_text"] for doc in inserted] == ["Keep this", "Also keep"]


def test_create_embeddings_rejects_json_array_source_query(client, mock_clients) -> None:
    """A JSON-encoded array is not a query object and must 400 before find()."""
    dbs = _wire_mongo(mock_clients["mongo"])
    source_collection = dbs["source_db"].__getitem__.return_value
    response = client.post(
        "/create-embeddings",
        json={
            "source_database_name": "src",
            "source_collection_name": "docs",
            "index_database_name": "index_db",
            "index_collection_name": "chunks",
            "source_query": "[1, 2]",
        },
    )
    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]
    source_collection.find.assert_not_called()
