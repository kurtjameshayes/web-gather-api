"""Regression tests for /create-embeddings default text_column and query guards.

Omitting text_column currently defaults to chunk_text. Dangerous Mongo
operators in source_query are rejected before any collection read.
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

from core import core_bp, init_core


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in inputs])


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients) -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_source_and_index(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_coll = MagicMock(name="source_coll")
    index_coll = MagicMock(name="index_coll")
    source_db = MagicMock(name="source_db")
    index_db = MagicMock(name="index_db")
    source_db.__getitem__.return_value = source_coll
    index_db.__getitem__.return_value = index_coll

    def _get_db(name: str) -> MagicMock:
        return index_db if name == "index_db" else source_db

    mock_mongo.__getitem__.side_effect = _get_db
    return {"source_coll": source_coll, "index_coll": index_coll}


def test_create_embeddings_omitted_text_column_defaults_to_chunk_text(
    client, mock_clients
) -> None:
    """Omitting text_column embeds the chunk_text field and echoes that default."""
    colls = _wire_source_and_index(mock_clients["mongo"])
    colls["source_coll"].find.return_value = [
        {"_id": "row-1", "chunk_text": "Indexed body"},
    ]
    colls["index_coll"].delete_many.return_value.deleted_count = 0

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
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
    assert payload["text_column"] == "chunk_text"
    assert payload["chunks_indexed"] == 1
    colls["source_coll"].find.assert_called_once_with({})
    inserted = colls["index_coll"].insert_many.call_args[0][0]
    assert inserted[0]["chunk_text"] == "Indexed body"
    assert inserted[0]["source_id"] == "row-1"


def test_create_embeddings_rejects_where_operator_before_mongo_read(
    client, mock_clients
) -> None:
    """source_query $where is a 400 NoSQL-injection guard and must not call find()."""
    colls = _wire_source_and_index(mock_clients["mongo"])

    response = client.post(
        "/create-embeddings",
        json={
            "source_database_name": "src",
            "source_collection_name": "docs",
            "index_database_name": "index_db",
            "index_collection_name": "chunks",
            "source_query": {"$where": "this.active === true"},
        },
    )

    assert response.status_code == 400
    assert "not allowed" in response.get_json()["error"]
    colls["source_coll"].find.assert_not_called()
    colls["index_coll"].insert_many.assert_not_called()
    colls["index_coll"].delete_many.assert_not_called()
