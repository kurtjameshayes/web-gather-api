"""Regression tests for GET /search index-location fallbacks.

When document metadata omits index_database_name / chunk_collection, search
must read chunks from the source database/collection and look up the embedding
model by that source database. Base tests always set the index fields, so a
regression here would query the wrong collection or skip configured models.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import WEB_GATHER_DB, core_bp, init_core  # noqa: E402


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


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
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


def _wire_source_and_catalog(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_db = MagicMock(name="source_db")
    source_coll = MagicMock(name="source_coll")
    source_db.__getitem__.return_value = source_coll
    wg_db = MagicMock(name="wg_db")
    docs_coll = MagicMock(name="docs_coll")
    wg_db.__getitem__.return_value = docs_coll

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    return {
        "source_db": source_db,
        "source_coll": source_coll,
        "docs_coll": docs_coll,
    }


def test_search_falls_back_to_source_database_and_collection(
    client, mock_clients
) -> None:
    """Missing index metadata uses database_name/collection_name for chunks and model."""
    dbs = _wire_source_and_catalog(mock_clients["mongo"])
    dbs["docs_coll"].find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "src",
        "collection_name": "docs",
    }
    dbs["source_coll"].find.return_value = [
        {"text": "retention policy", "embedding": [1.0, 0.0]},
    ]

    with patch("core.get_embedding_model_name", return_value="all-MiniLM-L6-v2") as model_lookup, patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.get("/search?document_id=doc-1&query=retention")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_id"] == "doc-1"
    assert payload["query"] == "retention"
    assert payload["results"][0]["text"] == "retention policy"
    model_lookup.assert_called_once_with("src")
    dbs["source_db"].__getitem__.assert_called_with("docs")
    dbs["source_coll"].find.assert_called_once_with({"document_id": "doc-1"}, {"_id": 0})
    dbs["docs_coll"].find_one.assert_called_once_with({"document_id": "doc-1"})


def test_search_uses_chunk_collection_override_on_source_database(
    client, mock_clients
) -> None:
    """chunk_collection can override the collection while still using source database."""
    dbs = _wire_source_and_catalog(mock_clients["mongo"])
    dbs["docs_coll"].find_one.return_value = {
        "document_id": "doc-2",
        "database_name": "src",
        "collection_name": "docs",
        "chunk_collection": "chunks",
    }
    dbs["source_coll"].find.return_value = [
        {"text": "opt out", "embedding": [1.0, 0.0]},
    ]

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.get("/search?document_id=doc-2&query=opt+out")

    assert response.status_code == 200
    dbs["source_db"].__getitem__.assert_called_with("chunks")
    assert response.get_json()["results"][0]["text"] == "opt out"
