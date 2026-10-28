"""Regression tests for shared `_parse_safe_query` decoding.

GET/DELETE /documents and category-mapping all parse the `query` argument twice
when the first JSON value is a string, then reject non-objects and operators.
A missed second decode turns a selective filter into a 400, or lets a
double-encoded operator through if checks run too early. A non-object that
falls through to `{}` would wipe a collection on DELETE.
"""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

from db import WEB_GATHER_DB, db_bp, init_db

_OID = "507f1f77bcf86cd799439011"


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_mongo_client() -> MagicMock:
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, Any]:
    wg_db = MagicMock()
    user_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.return_value = MagicMock()
    user_db.__getitem__.return_value = MagicMock()
    return {"wg_db": wg_db, "user_db": user_db}


def _docs_collection(mock_mongo: MagicMock):
    dbs = _wire_mongo(mock_mongo)
    coll = dbs["user_db"].__getitem__.return_value
    cursor = MagicMock()
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    coll.find.return_value = cursor
    coll.delete_many.return_value.deleted_count = 0
    return coll


def _twice(value: Any) -> str:
    """JSON-encode value, then encode that text as a JSON string."""
    return json.dumps(json.dumps(value))


def test_get_documents_unwraps_double_encoded_object(client, mock_mongo_client) -> None:
    """A JSON string of an object is the find filter, not a type error."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": _twice({"status": "inactive", "n": 0}),
        },
    )
    assert response.status_code == 200
    assert response.get_json()["documents"] == []
    coll.find.assert_called_once_with({"status": "inactive", "n": 0})
    coll.find.return_value.skip.assert_called_once_with(0)
    coll.find.return_value.limit.assert_called_once_with(100)


def test_delete_documents_unwraps_double_encoded_object(client, mock_mongo_client) -> None:
    """DELETE uses the inner object. It must not fall through to delete_many({})."""
    coll = _docs_collection(mock_mongo_client)
    coll.delete_many.return_value.deleted_count = 2
    response = client.delete(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": _twice({"status": "inactive"}),
        },
    )
    assert response.status_code == 200
    assert response.get_json()["deleted_count"] == 2
    coll.delete_many.assert_called_once_with({"status": "inactive"})


@pytest.mark.parametrize(
    "query",
    [
        _twice([1, 2]),
        json.dumps(5),
        json.dumps(None),
        json.dumps(True),
        _twice("5"),
    ],
)
def test_get_documents_rejects_double_decoded_non_object(
    client, mock_mongo_client, query: str
) -> None:
    """Non-objects stay 400 after the second decode and never reach find()."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": query,
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "query must be a JSON object"
    coll.find.assert_not_called()


def test_delete_documents_rejects_double_encoded_list(client, mock_mongo_client) -> None:
    """A double-encoded list must not be treated as an empty filter wipe."""
    coll = _docs_collection(mock_mongo_client)
    response = client.delete(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": _twice(["status"]),
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "query must be a JSON object"
    coll.delete_many.assert_not_called()


def test_second_decode_invalid_json_is_generic(client, mock_mongo_client) -> None:
    """A JSON string whose contents are not JSON is the generic invalid-JSON 400."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": json.dumps("hello"),
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid JSON in query parameter"
    coll.find.assert_not_called()


def test_double_encoded_operator_rejected_after_unwrap(client, mock_mongo_client) -> None:
    """$where hidden inside a JSON string is rejected after the second decode."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": _twice({"$where": "1==1"}),
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Query operator '$where' is not allowed"
    coll.find.assert_not_called()


def test_operator_nested_in_list_rejected(client, mock_mongo_client) -> None:
    """Operators nested under an array are rejected before delete_many."""
    coll = _docs_collection(mock_mongo_client)
    response = client.delete(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": json.dumps({"tags": [{"$regex": "^a"}]}),
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Query operator '$regex' is not allowed"
    coll.delete_many.assert_not_called()


def test_get_documents_converts_nested_oid(client, mock_mongo_client) -> None:
    """Extended JSON $oid values become ObjectId, including inside a list."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": json.dumps({"ids": [{"$oid": _OID}]}),
        },
    )
    assert response.status_code == 200
    coll.find.assert_called_once_with({"ids": [ObjectId(_OID)]})


def test_double_encoded_oid_converted_after_unwrap(client, mock_mongo_client) -> None:
    """$oid conversion runs on the object produced by the second decode."""
    coll = _docs_collection(mock_mongo_client)
    response = client.delete(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": _twice({"_id": {"$oid": _OID}}),
        },
    )
    assert response.status_code == 200
    coll.delete_many.assert_called_once_with({"_id": ObjectId(_OID)})


def test_invalid_oid_left_as_document(client, mock_mongo_client) -> None:
    """An invalid $oid is passed through unchanged and still queried."""
    coll = _docs_collection(mock_mongo_client)
    raw = {"_id": {"$oid": "not-an-object-id"}}
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": json.dumps(raw),
        },
    )
    assert response.status_code == 200
    coll.find.assert_called_once_with(raw)


def test_oid_with_sibling_key_rejected(client, mock_mongo_client) -> None:
    """$oid is only legal as the sole key of its object."""
    coll = _docs_collection(mock_mongo_client)
    response = client.get(
        "/documents",
        query_string={
            "database_name": "test_db",
            "collection_name": "test_collection",
            "query": json.dumps({"$oid": _OID, "extra": 1}),
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "$oid must be the only key in object"
    coll.find.assert_not_called()


def test_category_mapping_query_replaces_structured_filters(client, mock_mongo_client) -> None:
    """A query argument replaces statute_category and sub_topic instead of ANDing."""
    pc_db = MagicMock()
    coll = MagicMock()
    coll.find.return_value = []
    pc_db.__getitem__.return_value = coll
    mock_mongo_client.__getitem__.return_value = pc_db

    response = client.get(
        "/category-mapping",
        query_string={
            "statute_category": "sale",
            "sub_topic": "opt_out",
            "query": json.dumps({"document_id": "doc-9"}),
        },
    )
    assert response.status_code == 200
    coll.find.assert_called_once_with({"document_id": "doc-9"})


def test_category_mapping_empty_query_drops_statute_filter(client, mock_mongo_client) -> None:
    """Double-encoded {} wins over statute_category and scans every mapping."""
    pc_db = MagicMock()
    coll = MagicMock()
    coll.find.return_value = []
    pc_db.__getitem__.return_value = coll
    mock_mongo_client.__getitem__.return_value = pc_db

    response = client.get(
        "/category-mapping",
        query_string={
            "statute_category": "sale",
            "query": _twice({}),
        },
    )
    assert response.status_code == 200
    coll.find.assert_called_once_with({})


def test_category_mapping_invalid_query_does_not_fall_back(client, mock_mongo_client) -> None:
    """Invalid query is 400 even when statute_category could have filtered."""
    pc_db = MagicMock()
    coll = MagicMock()
    pc_db.__getitem__.return_value = coll
    mock_mongo_client.__getitem__.return_value = pc_db

    response = client.get(
        "/category-mapping",
        query_string={"statute_category": "sale", "query": "{bad"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid JSON in query parameter"
    coll.find.assert_not_called()


def test_delete_category_mapping_double_encoded_list_does_not_delete(
    client, mock_mongo_client
) -> None:
    """A non-object query does not delete category mappings."""
    pc_db = MagicMock()
    coll = MagicMock()
    pc_db.__getitem__.return_value = coll
    mock_mongo_client.__getitem__.return_value = pc_db

    response = client.delete(
        "/category-mapping",
        query_string={"query": _twice([{"statute_category": "sale"}])},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "query must be a JSON object"
    coll.delete_many.assert_not_called()


def test_delete_category_mapping_unwraps_double_encoded_filter(
    client, mock_mongo_client
) -> None:
    """DELETE /category-mapping applies the inner object after the second decode."""
    pc_db = MagicMock()
    coll = MagicMock()
    coll.delete_many.return_value.deleted_count = 1
    pc_db.__getitem__.return_value = coll
    mock_mongo_client.__getitem__.return_value = pc_db

    response = client.delete(
        "/category-mapping",
        query_string={"query": _twice({"statute_category": "sale"})},
    )
    assert response.status_code == 200
    assert response.get_json()["deleted_count"] == 1
    coll.delete_many.assert_called_once_with({"statute_category": "sale"})
