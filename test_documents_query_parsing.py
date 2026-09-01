"""Query-parameter parsing contracts for GET/DELETE /documents."""
from __future__ import annotations

import json
import urllib.parse
from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import WEB_GATHER_DB, db_bp, init_db


@pytest.fixture
def mock_mongo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(mock_mongo: MagicMock):
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def _wire_user_collection(mock_mongo: MagicMock) -> MagicMock:
    wg_db = MagicMock()
    user_db = MagicMock()
    collection = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    user_db.__getitem__.return_value = collection
    cursor = MagicMock()
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    collection.find.return_value = cursor
    return collection


def test_documents_double_encoded_json_query_is_decoded(client, mock_mongo) -> None:
    """A JSON string whose value is itself a JSON object is decoded twice.

    `_parse_safe_query` json.loads, then json.loads again when the first result is a str.
    """
    collection = _wire_user_collection(mock_mongo)
    inner = json.dumps({"status": "active"})
    query = urllib.parse.quote(json.dumps(inner))
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 200
    collection.find.assert_called_once_with({"status": "active"})


def test_documents_query_json_array_is_400(client, mock_mongo) -> None:
    query = urllib.parse.quote("[1, 2]")
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "query must be a JSON object"
    mock_mongo.__getitem__.assert_not_called()


def test_documents_oid_with_extra_keys_is_400(client, mock_mongo) -> None:
    query = urllib.parse.quote(
        '{"$oid": "64b8f7b7b1f1eaf5b2f0c001", "extra": true}'
    )
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 400
    assert "$oid must be the only key" in response.get_json()["error"]


def test_get_collections_missing_database_name_is_400(client, mock_mongo) -> None:
    response = client.get("/collections")
    assert response.status_code == 400
    assert response.get_json()["error"] == "database_name is required"


def test_get_all_collections_missing_database_name_is_400(client, mock_mongo) -> None:
    response = client.get("/all-collections")
    assert response.status_code == 400
    assert response.get_json()["error"] == "database_name is required"
