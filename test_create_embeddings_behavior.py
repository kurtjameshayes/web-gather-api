"""Regression tests for /create-embeddings write semantics and model selection.

Existing tests cover missing params, invalid source_query, and status counts.
These tests assert scoped deletes, insert payloads, in-place updates, empty-row
skips, and privacy-compliance application-model preference.
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

from core import PRIVACY_COMPLIANCE_DB, core_bp, init_core
from db import get_application_embedding_model, set_application_embedding_model


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[float(i + 1), 0.0] for i in range(len(inputs))])


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


@pytest.fixture(autouse=True)
def reset_application_embedding_model():
    previous = get_application_embedding_model()
    set_application_embedding_model(None)
    yield
    set_application_embedding_model(previous)


def _wire_distinct_collections(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    """Wire source/index DBs with distinct collection mocks for write assertions."""
    source_db = MagicMock(name="source_db")
    index_db = MagicMock(name="index_db")
    source_coll = MagicMock(name="source_coll")
    index_coll = MagicMock(name="index_coll")
    source_db.__getitem__.return_value = source_coll
    index_db.__getitem__.return_value = index_coll

    def _get_db(name: str) -> MagicMock:
        if name in {"index_db", PRIVACY_COMPLIANCE_DB}:
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    return {
        "source_db": source_db,
        "index_db": index_db,
        "source_coll": source_coll,
        "index_coll": index_coll,
    }


def test_create_embeddings_cross_collection_scopes_delete_and_inserts_payload(
    client, mock_clients
) -> None:
    dbs = _wire_distinct_collections(mock_clients["mongo"])
    dbs["source_coll"].find.return_value = [
        {
            "_id": "row-1",
            "document_id": "doc-1",
            "chunk_text": "First chunk",
            "category": "retention",
        },
        {"_id": "row-2", "document_id": "doc-1", "chunk_text": "   "},
        {"_id": "row-3", "document_id": "doc-1", "chunk_text": "Second chunk"},
    ]
    dbs["index_coll"].delete_many.return_value = MagicMock(deleted_count=2)

    with patch("core.get_embedding_model_name", return_value="model-a"), patch(
        "core.get_model", return_value=DummyModel()
    ), patch("core.utc_now", return_value="2026-08-08T00:00:00+00:00"):
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "chunks",
                "index_database_name": "index_db",
                "index_collection_name": "embeddings",
                "source_query": {"document_id": "doc-1"},
                "text_column": "chunk_text",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunks_indexed"] == 2
    assert payload["skipped_rows"] == 1
    assert payload["embedding_model"] == "model-a"
    assert payload["text_column"] == "chunk_text"

    dbs["source_coll"].find.assert_called_once_with({"document_id": "doc-1"})
    dbs["index_coll"].delete_many.assert_called_once_with(
        {
            "source_database_name": "src",
            "source_collection_name": "chunks",
            "document_id": "doc-1",
        }
    )
    dbs["index_coll"].insert_many.assert_called_once()
    inserted = dbs["index_coll"].insert_many.call_args.args[0]
    assert len(inserted) == 2
    assert inserted[0]["source_id"] == "row-1"
    assert inserted[0]["chunk_text"] == "First chunk"
    assert inserted[0]["embedding"] == [1.0, 0.0]
    assert inserted[0]["category"] == "retention"
    assert inserted[0]["source_database_name"] == "src"
    assert inserted[0]["source_collection_name"] == "chunks"
    assert inserted[0]["indexed_at"] == "2026-08-08T00:00:00+00:00"
    assert inserted[1]["source_id"] == "row-3"
    assert inserted[1]["embedding"] == [2.0, 0.0]
    assert "_id" not in inserted[0]
    dbs["index_coll"].update_one.assert_not_called()


def test_create_embeddings_same_collection_updates_in_place(client, mock_clients) -> None:
    source_db = MagicMock()
    coll = MagicMock()
    source_db.__getitem__.return_value = coll
    mock_clients["mongo"].__getitem__.return_value = source_db
    coll.find.return_value = [
        {"_id": "same-1", "sub_chunk_text": "In-place text", "category": "notice"},
    ]

    with patch("core.get_embedding_model_name", return_value="model-b"), patch(
        "core.get_model", return_value=DummyModel()
    ), patch("core.utc_now", return_value="ts"):
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "same_db",
                "source_collection_name": "same_coll",
                "index_database_name": "same_db",
                "index_collection_name": "same_coll",
                "text_column": "sub_chunk_text",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["chunks_indexed"] == 1
    coll.delete_many.assert_not_called()
    coll.insert_many.assert_not_called()
    coll.update_one.assert_called_once_with(
        {"_id": "same-1"},
        {
            "$set": {
                "embedding": [1.0, 0.0],
                "sub_chunk_text": "In-place text",
                "indexed_at": "ts",
                "source_id": "same-1",
                "source_database_name": "same_db",
                "source_collection_name": "same_coll",
                "category": "notice",
            }
        },
    )


def test_create_embeddings_privacy_compliance_prefers_application_model(
    client, mock_clients
) -> None:
    dbs = _wire_distinct_collections(mock_clients["mongo"])
    dbs["source_coll"].find.return_value = [
        {"_id": "p1", "chunk_text": "Privacy chunk"},
    ]
    dbs["index_coll"].delete_many.return_value = MagicMock(deleted_count=0)
    set_application_embedding_model("app-default-model")

    with patch("core.get_embedding_model_name", return_value="db-model") as get_db_model, patch(
        "core.get_model", return_value=DummyModel()
    ) as get_model:
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "policy_chunks",
                "index_database_name": PRIVACY_COMPLIANCE_DB,
                "index_collection_name": "policy_legal_embeddings",
                "text_column": "chunk_text",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["embedding_model"] == "app-default-model"
    get_db_model.assert_not_called()
    get_model.assert_called_once_with("app-default-model")


def test_create_embeddings_rejects_blank_text_column_and_all_empty_rows(
    client, mock_clients
) -> None:
    response = client.post(
        "/create-embeddings",
        json={
            "source_database_name": "src",
            "source_collection_name": "chunks",
            "index_database_name": "index_db",
            "index_collection_name": "embeddings",
            "text_column": "   ",
        },
    )
    assert response.status_code == 400
    assert "text_column" in response.get_json()["error"]

    dbs = _wire_distinct_collections(mock_clients["mongo"])
    dbs["source_coll"].find.return_value = [
        {"_id": "e1", "chunk_text": ""},
        {"_id": "e2", "chunk_text": "   "},
    ]
    with patch("core.get_embedding_model_name", return_value="model-a"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "chunks",
                "index_database_name": "index_db",
                "index_collection_name": "embeddings",
            },
        )
    assert response.status_code == 400
    assert "no rows" in response.get_json()["error"]
    dbs["index_coll"].insert_many.assert_not_called()
