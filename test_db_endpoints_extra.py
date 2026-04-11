"""Additional tests for database endpoints."""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

import security
from db import DOCUMENTS_COLLECTION, WEB_GATHER_DB, db_bp, init_db


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


def test_list_documents_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    first_id = ObjectId()
    second_id = ObjectId()
    docs = [{"_id": first_id, "name": "doc1"}, {"_id": second_id, "name": "doc2"}]
    cursor_mock = MagicMock()
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = docs
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor_mock
    response = client.get(
        "/documents?database_name=test_db&collection_name=test_collection"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["_id"] == str(first_id)
    assert payload["documents"][1]["_id"] == str(second_id)


def test_list_documents_invalid_query_json(client, mock_mongo_client) -> None:
    response = client.get(
        "/documents?database_name=test_db&collection_name=test_collection&query={bad}"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "Invalid JSON" in payload["error"]


def test_list_documents_rejects_dangerous_operators(client, mock_mongo_client) -> None:
    """NoSQL injection: $where, $regex, etc. must be rejected."""
    import urllib.parse
    query = urllib.parse.quote('{"$where": "1==1"}')
    response = client.get(
        f"/documents?database_name=test_db&collection_name=test_collection&query={query}"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "not allowed" in payload["error"].lower() or "operator" in payload["error"].lower()


def test_list_documents_invalid_collection_name(client, mock_mongo_client) -> None:
    """Invalid chars in database/collection name must be rejected."""
    response = client.get(
        "/documents?database_name=test.db&collection_name=my_collection"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "letters, numbers" in payload["error"] or "invalid" in payload["error"].lower()


def test_list_documents_reserved_database(client, mock_mongo_client) -> None:
    """Reserved database names (admin, config, local) must be rejected."""
    response = client.get(
        "/documents?database_name=admin&collection_name=users"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "reserved" in payload["error"].lower()


def test_delete_documents_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    delete_result = MagicMock()
    delete_result.deleted_count = 3
    dbs["user_db"].__getitem__.return_value.delete_many.return_value = delete_result
    response = client.delete(
        "/documents?database_name=test_db&collection_name=test_collection"
        "&query=%7B%22status%22%3A%20%22inactive%22%7D"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted_count"] == 3
    dbs["user_db"].__getitem__.return_value.delete_many.assert_called_with(
        {"status": "inactive"}
    )


def test_delete_documents_invalid_query_json(client, mock_mongo_client) -> None:
    response = client.delete(
        "/documents?database_name=test_db&collection_name=test_collection&query={bad}"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "Invalid JSON" in payload["error"]


def test_delete_documents_rejects_dangerous_operators(client, mock_mongo_client) -> None:
    """NoSQL injection: $where must be rejected on DELETE."""
    import urllib.parse
    query = urllib.parse.quote('{"$where": "1==1"}')
    response = client.delete(
        f"/documents?database_name=test_db&collection_name=test_collection&query={query}"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "not allowed" in payload["error"].lower() or "operator" in payload["error"].lower()


def test_delete_documents_missing_params(client, mock_mongo_client) -> None:
    response = client.delete("/documents")
    assert response.status_code == 400
    payload = response.get_json()
    assert "database_name" in payload["error"]


def test_list_collections_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    dbs["wg_db"].__getitem__.return_value.distinct.return_value = [
        "docs",
        "policies",
    ]
    response = client.get("/collections?database_name=test_db")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["collections"] == ["docs", "policies"]


def test_list_databases_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    dbs["wg_db"].__getitem__.return_value.distinct.return_value = ["db1", "db2"]
    response = client.get("/databases")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["databases"] == ["db1", "db2"]


def test_list_all_databases_success(client, mock_mongo_client) -> None:
    mock_mongo_client.list_database_names.return_value = ["db1", "db2"]
    response = client.get("/all-databases")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["databases"] == ["db1", "db2"]


def test_list_all_databases_filters_reserved_names(client, mock_mongo_client) -> None:
    mock_mongo_client.list_database_names.return_value = [
        "db1",
        "admin",
        "config",
        "local",
        "db2",
    ]
    response = client.get("/all-databases")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["databases"] == ["db1", "db2"]


def test_list_all_collections_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].list_collection_names.return_value = ["a", "b"]
    response = client.get("/all-collections?database_name=test_db")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["collections"] == ["a", "b"]


def test_list_all_collections_rejects_reserved_database(client, mock_mongo_client) -> None:
    response = client.get("/all-collections?database_name=admin")
    assert response.status_code == 400
    payload = response.get_json()
    assert "reserved" in payload["error"].lower()


def test_write_to_collection_append_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].__getitem__.return_value.insert_one.return_value.inserted_id = "id1"
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "append",
            "document": {"name": "doc1"},
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "append"


def test_write_to_collection_document_must_be_object(client, mock_mongo_client) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "append",
            "document": ["not", "an", "object"],
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "json object" in payload["error"].lower()


def test_write_to_collection_rejects_reserved_database(client, mock_mongo_client) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "admin",
            "collection_name": "docs",
            "mode": "append",
            "document": {"name": "doc1"},
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "reserved" in payload["error"].lower()


def test_documents_requires_api_key_when_configured(client, mock_mongo_client, monkeypatch) -> None:
    monkeypatch.setattr(security, "_app_api_key", "secret")
    response = client.get("/documents?database_name=test_db&collection_name=test_collection")
    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"] == "Missing API key."


def test_documents_rejects_wrong_api_key_when_configured(client, mock_mongo_client, monkeypatch) -> None:
    monkeypatch.setattr(security, "_app_api_key", "secret")
    response = client.get(
        "/documents?database_name=test_db&collection_name=test_collection",
        headers={"x-api-key": "wrong"},
    )
    assert response.status_code == 403
    payload = response.get_json()
    assert payload["error"] == "Invalid API key."


def test_documents_allows_valid_api_key_when_configured(client, mock_mongo_client, monkeypatch) -> None:
    monkeypatch.setattr(security, "_app_api_key", "secret")
    dbs = _wire_mongo(mock_mongo_client)
    cursor_mock = MagicMock()
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = []
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor_mock
    response = client.get(
        "/documents?database_name=test_db&collection_name=test_collection",
        headers={"x-api-key": "secret"},
    )
    assert response.status_code == 200


def test_write_to_collection_replace_invalid_id(client, mock_mongo_client) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "document": {"name": "doc1"},
            "update_id": "not-an-objectid",
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "Invalid update_id" in payload["error"]


def test_write_to_collection_replace_not_found(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].__getitem__.return_value.replace_one.return_value.matched_count = 0
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "document": {"name": "doc1"},
            "update_id": "64b8f7b7b1f1eaf5b2f0c001",
        },
    )
    assert response.status_code == 404
    payload = response.get_json()
    assert "not found" in payload["error"].lower()
