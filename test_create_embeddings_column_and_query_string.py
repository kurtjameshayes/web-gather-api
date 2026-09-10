"""POST /create-embeddings text_column defaulting and JSON-string source_query types."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from conftest import _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


_BASE = {
    "source_database_name": "src",
    "source_collection_name": "docs",
    "index_database_name": "index_db",
    "index_collection_name": "chunks",
}


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in inputs])


def test_json_string_list_source_query_is_400(client, mock_clients) -> None:
    """A JSON-string array is parsed then rejected as a non-object (distinct from a list-typed body)."""
    _wire_mongo_core(mock_clients["mongo"])
    response = client.post(
        "/create-embeddings",
        json={**_BASE, "source_query": "[1, 2]"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "source_query must be a JSON object"


def test_empty_text_column_defaults_to_chunk_text(client, mock_clients) -> None:
    """Empty string is falsy so `or 'chunk_text'` applies; whitespace-only is rejected later."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [{"_id": "row-1", "chunk_text": "Sample text"}]

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post("/create-embeddings", json={**_BASE, "text_column": ""})
    assert response.status_code == 200
    assert response.get_json()["text_column"] == "chunk_text"


def test_whitespace_text_column_is_400(client, mock_clients) -> None:
    """Whitespace-only text_column is truthy, then fails the non-empty strip check."""
    _wire_mongo_core(mock_clients["mongo"])
    response = client.post("/create-embeddings", json={**_BASE, "text_column": "   "})
    assert response.status_code == 400
    assert response.get_json()["error"] == "text_column must be a non-empty string"
