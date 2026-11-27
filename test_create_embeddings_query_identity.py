"""Regression tests for /create-embeddings query identity and value coercion.

Existing coverage checks a document_id-scoped delete, in-place updates, and
blank-text rejection. These tests cover the delete filter taking source_query
keys over the request's source identity, $oid filters staying raw dicts, and
falsy text values that still get indexed, including a null category copied from the source row.
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

from core import core_bp, init_core


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[float(i + 1), 0.0] for i in range(len(inputs))])


@pytest.fixture
def client() -> Any:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    app.config["mongo"] = mock_mongo
    return app.test_client()


def _mongo(client) -> MagicMock:
    return client.application.config["mongo"]


def _wire(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_db = MagicMock(name="source_db")
    index_db = MagicMock(name="index_db")
    source_coll = MagicMock(name="source_coll")
    index_coll = MagicMock(name="index_coll")
    source_db.__getitem__.return_value = source_coll
    index_db.__getitem__.return_value = index_coll

    def _get_db(name: str) -> MagicMock:
        if name == "index_db":
            return index_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    index_coll.delete_many.return_value = MagicMock(deleted_count=0)
    return {"source_coll": source_coll, "index_coll": index_coll}


def _post(client, body: Dict[str, Any]):
    with patch("core.get_embedding_model_name", return_value="model-a"), patch(
        "core.get_model", return_value=DummyModel()
    ), patch("core.utc_now", return_value="2026-11-27T00:00:00+00:00"):
        return client.post("/create-embeddings", json=body)


def _base_body(**overrides: Any) -> Dict[str, Any]:
    body = {
        "source_database_name": "src",
        "source_collection_name": "chunks",
        "index_database_name": "index_db",
        "index_collection_name": "embeddings",
        "text_column": "chunk_text",
    }
    body.update(overrides)
    return body


def test_source_query_overwrites_delete_identity_but_insert_keeps_request_names(client) -> None:
    dbs = _wire(_mongo(client))
    dbs["source_coll"].find.return_value = [
        {
            "_id": "row-1",
            "chunk_text": "hello",
            "document_id": "doc-9",
            "source_database_name": "from-doc",
            "source_collection_name": "from-doc-coll",
        }
    ]
    source_query = {
        "source_database_name": "other-db",
        "source_collection_name": "other-coll",
        "document_id": "doc-9",
    }

    response = _post(client, _base_body(source_query=source_query))

    assert response.status_code == 200
    dbs["source_coll"].find.assert_called_once_with(source_query)
    dbs["index_coll"].delete_many.assert_called_once_with(
        {
            "source_database_name": "other-db",
            "source_collection_name": "other-coll",
            "document_id": "doc-9",
        }
    )
    inserted = dbs["index_coll"].insert_many.call_args.args[0]
    assert inserted[0]["source_database_name"] == "src"
    assert inserted[0]["source_collection_name"] == "chunks"
    assert inserted[0]["source_id"] == "row-1"
    assert inserted[0]["document_id"] == "doc-9"
    assert inserted[0]["chunk_text"] == "hello"


def test_oid_filter_is_passed_through_without_objectid_conversion(client) -> None:
    dbs = _wire(_mongo(client))
    dbs["source_coll"].find.return_value = [
        {"_id": "507f1f77bcf86cd799439011", "chunk_text": "statute text"}
    ]
    oid_query = {"_id": {"$oid": "507f1f77bcf86cd799439011"}}

    response = _post(client, _base_body(source_query=oid_query))

    assert response.status_code == 200
    dbs["source_coll"].find.assert_called_once_with(oid_query)
    delete_filter = dbs["index_coll"].delete_many.call_args.args[0]
    assert delete_filter["_id"] == {"$oid": "507f1f77bcf86cd799439011"}
    assert isinstance(delete_filter["_id"], dict)
    assert delete_filter["source_database_name"] == "src"
    assert delete_filter["source_collection_name"] == "chunks"


def test_malformed_source_query_is_rejected_before_find(client) -> None:
    dbs = _wire(_mongo(client))
    dbs["source_coll"].find.return_value = [{"_id": "row-1", "chunk_text": "kept"}]

    sibling = _post(
        client,
        _base_body(source_query={"_id": {"$oid": "507f1f77bcf86cd799439011", "extra": 1}}),
    )
    assert sibling.status_code == 400
    assert "$oid must be the only key" in sibling.get_json()["error"]

    non_object = _post(client, _base_body(source_query=["document_id"]))
    assert non_object.status_code == 400
    assert non_object.get_json()["error"] == "source_query must be a JSON object"

    empty = _post(client, _base_body(source_query=""))
    assert empty.status_code == 400
    assert "Invalid JSON in source_query" in empty.get_json()["error"]

    whitespace = _post(client, _base_body(source_query="   "))
    assert whitespace.status_code == 400
    assert "Invalid JSON in source_query" in whitespace.get_json()["error"]

    dbs["source_coll"].find.assert_not_called()
    dbs["index_coll"].delete_many.assert_not_called()


def test_falsy_text_and_category_values_are_indexed(client) -> None:
    dbs = _wire(_mongo(client))
    dbs["source_coll"].find.return_value = [
        {"_id": "n0", "chunk_text": 0, "category": 0},
        {"_id": "bool", "chunk_text": False, "category": ""},
        {"_id": "blank", "chunk_text": "   ", "category": "skip-me"},
        {"_id": "kept", "chunk_text": "kept", "category": None},
        {"_id": "nocat", "chunk_text": "no category key"},
    ]

    response = _post(client, _base_body(text_column="  chunk_text  "))

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["text_column"] == "chunk_text"
    assert payload["chunks_indexed"] == 4
    assert payload["skipped_rows"] == 1

    inserted = dbs["index_coll"].insert_many.call_args.args[0]
    by_id = {doc["source_id"]: doc for doc in inserted}
    assert by_id["n0"]["chunk_text"] == "0"
    assert by_id["n0"]["category"] == 0
    assert by_id["bool"]["chunk_text"] == "False"
    assert by_id["bool"]["category"] == ""
    # Source keys are copied before the category guard, so an explicit null stays.
    assert by_id["kept"]["category"] is None
    assert by_id["kept"]["chunk_text"] == "kept"
    assert "category" not in by_id["nocat"]
    assert "blank" not in by_id
