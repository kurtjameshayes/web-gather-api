"""Regression tests for GET /documents pagination and required params.

Route-level limit/offset clamps were untested on base and are distinct from
compliance GET /runs and /alerts pagination coverage.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

from db import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, WEB_GATHER_DB, db_bp, init_db


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


def _cursor_with_docs(docs: list[dict]) -> MagicMock:
    cursor = MagicMock()
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def test_list_documents_defaults_limit_and_offset(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    doc_id = ObjectId()
    cursor = _cursor_with_docs([{"_id": doc_id, "name": "doc1"}])
    collection = dbs["user_db"].__getitem__.return_value
    collection.find.return_value = cursor

    response = client.get("/documents?database_name=test_db&collection_name=docs")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == DEFAULT_PAGE_SIZE
    assert payload["offset"] == 0
    assert payload["documents"][0]["_id"] == str(doc_id)
    cursor.skip.assert_called_once_with(0)
    cursor.limit.assert_called_once_with(DEFAULT_PAGE_SIZE)


def test_list_documents_clamps_limit_to_max_page_size(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor

    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&limit={MAX_PAGE_SIZE + 5000}"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == MAX_PAGE_SIZE
    cursor.limit.assert_called_once_with(MAX_PAGE_SIZE)


def test_list_documents_clamps_non_positive_limit_to_one(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor

    response = client.get(
        "/documents?database_name=test_db&collection_name=docs&limit=0"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 1
    cursor.limit.assert_called_once_with(1)


def test_list_documents_invalid_limit_falls_back_to_default(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor

    response = client.get(
        "/documents?database_name=test_db&collection_name=docs&limit=not-a-number"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == DEFAULT_PAGE_SIZE
    cursor.limit.assert_called_once_with(DEFAULT_PAGE_SIZE)


def test_list_documents_clamps_negative_offset_and_invalid_offset(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    collection = dbs["user_db"].__getitem__.return_value
    collection.find.return_value = cursor

    negative = client.get(
        "/documents?database_name=test_db&collection_name=docs&offset=-25"
    )
    assert negative.status_code == 200
    assert negative.get_json()["offset"] == 0
    cursor.skip.assert_called_with(0)

    cursor.reset_mock()
    invalid = client.get(
        "/documents?database_name=test_db&collection_name=docs&offset=abc"
    )
    assert invalid.status_code == 200
    assert invalid.get_json()["offset"] == 0
    cursor.skip.assert_called_with(0)


def test_list_documents_applies_requested_limit_and_offset(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor

    response = client.get(
        "/documents?database_name=test_db&collection_name=docs&limit=25&offset=50"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 25
    assert payload["offset"] == 50
    cursor.skip.assert_called_once_with(50)
    cursor.limit.assert_called_once_with(25)


def test_list_documents_missing_required_params(client, mock_mongo_client) -> None:
    missing_both = client.get("/documents")
    assert missing_both.status_code == 400
    assert "database_name" in missing_both.get_json()["error"]

    missing_collection = client.get("/documents?database_name=test_db")
    assert missing_collection.status_code == 400
    assert "collection_name" in missing_collection.get_json()["error"]
