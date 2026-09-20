"""Leftover create-embeddings index-write contracts: scoped wipe, same-target, $oid."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from conftest import DummyModel, _wire_mongo_core
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


def _index_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "source_database_name": "src",
        "source_collection_name": "docs",
        "index_database_name": "index_db",
        "index_collection_name": "chunks",
    }
    payload.update(overrides)
    return payload


def test_create_embeddings_delete_merges_source_query(client, mock_clients) -> None:
    """Different source/dest: delete_many is scoped to source_* plus source_query."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    index_coll = dbs["index_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text", "category": "consumer_rights"},
    ]
    index_coll.delete_many.return_value.deleted_count = 2

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/create-embeddings",
            json=_index_payload(source_query={"document_id": "doc-1"}),
        )

    assert response.status_code == 200
    source_coll.find.assert_called_once_with({"document_id": "doc-1"})
    index_coll.delete_many.assert_called_once_with(
        {
            "source_database_name": "src",
            "source_collection_name": "docs",
            "document_id": "doc-1",
        }
    )
    inserted = index_coll.insert_many.call_args[0][0]
    assert len(inserted) == 1
    assert inserted[0]["source_id"] == "row-1"
    assert inserted[0]["category"] == "consumer_rights"
    assert inserted[0]["source_database_name"] == "src"
    assert inserted[0]["source_collection_name"] == "docs"
    assert "_id" not in inserted[0]
    index_coll.update_one.assert_not_called()


def test_create_embeddings_delete_without_source_query_uses_source_fields_only(
    client, mock_clients
) -> None:
    """Omitted source_query must not wipe the destination with {}."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    index_coll = dbs["index_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text"},
    ]
    index_coll.delete_many.return_value.deleted_count = 0

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post("/create-embeddings", json=_index_payload())

    assert response.status_code == 200
    index_coll.delete_many.assert_called_once_with(
        {
            "source_database_name": "src",
            "source_collection_name": "docs",
        }
    )


def test_create_embeddings_same_target_updates_in_place(client, mock_clients) -> None:
    """Same source and index collection uses update_one and never delete_many."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text", "category": "controller_duties"},
    ]

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/create-embeddings",
            json=_index_payload(
                index_database_name="src",
                index_collection_name="docs",
            ),
        )

    assert response.status_code == 200
    coll.delete_many.assert_not_called()
    coll.insert_many.assert_not_called()
    coll.update_one.assert_called_once()
    selector, update = coll.update_one.call_args[0]
    assert selector == {"_id": "row-1"}
    set_fields = update["$set"]
    assert set_fields["source_id"] == "row-1"
    assert set_fields["category"] == "controller_duties"
    assert set_fields["source_database_name"] == "src"
    assert set_fields["chunk_text"] == "Sample text"
    assert set_fields["embedding"] == [1.0, 0.0]


def test_create_embeddings_oid_is_allowed_but_not_converted(client, mock_clients) -> None:
    """$oid passes the operator guard but is forwarded as a dict, not ObjectId."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    index_coll = dbs["index_db"].__getitem__.return_value
    oid_query = {"_id": {"$oid": "64b8f7b7b1f1eaf5b2f0c001"}}
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": "Sample text"},
    ]
    index_coll.delete_many.return_value.deleted_count = 0

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_model", return_value=DummyModel()
    ):
        response = client.post(
            "/create-embeddings",
            json=_index_payload(source_query=oid_query),
        )

    assert response.status_code == 200
    source_coll.find.assert_called_once_with(oid_query)
    delete_filter = index_coll.delete_many.call_args[0][0]
    assert delete_filter["_id"] == {"$oid": "64b8f7b7b1f1eaf5b2f0c001"}
    assert delete_filter["source_database_name"] == "src"


def test_create_embeddings_oid_with_extra_keys_rejected(client, mock_clients) -> None:
    """$oid must be the only key in its object; extra keys are 400 before find()."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value

    response = client.post(
        "/create-embeddings",
        json=_index_payload(
            source_query={"$oid": "64b8f7b7b1f1eaf5b2f0c001", "extra": 1},
        ),
    )

    assert response.status_code == 400
    assert "oid" in response.get_json()["error"].lower()
    source_coll.find.assert_not_called()
