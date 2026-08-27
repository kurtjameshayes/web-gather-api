"""GET/DELETE /documents Extended JSON $oid conversion through the HTTP query contract."""
from __future__ import annotations

import urllib.parse
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

from db import WEB_GATHER_DB, db_bp, init_db


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


def test_get_documents_converts_valid_oid_to_objectid(client, mock_mongo_client) -> None:
    """{"_id": {"$oid": "<hex>"}} is converted to ObjectId before find()."""
    dbs = _wire_mongo(mock_mongo_client)
    oid = ObjectId("64b8f7b7b1f1eaf5b2f0c001")
    cursor_mock = MagicMock()
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = [{"_id": oid, "name": "doc1"}]
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor_mock

    query = urllib.parse.quote('{"_id": {"$oid": "64b8f7b7b1f1eaf5b2f0c001"}}')
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["documents"][0]["_id"] == str(oid)

    find_query = dbs["user_db"].__getitem__.return_value.find.call_args.args[0]
    assert find_query == {"_id": oid}
    assert isinstance(find_query["_id"], ObjectId)


def test_get_documents_invalid_oid_is_left_unchanged(client, mock_mongo_client) -> None:
    """Invalid $oid values are passed through as the original dict (silent no-match)."""
    dbs = _wire_mongo(mock_mongo_client)
    cursor_mock = MagicMock()
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = []
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor_mock

    query = urllib.parse.quote('{"_id": {"$oid": "not-a-valid-oid"}}')
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 200
    find_query = dbs["user_db"].__getitem__.return_value.find.call_args.args[0]
    assert find_query == {"_id": {"$oid": "not-a-valid-oid"}}


def test_get_documents_oid_with_extra_keys_is_rejected(client, mock_mongo_client) -> None:
    """$oid must be the only key in its object; extra keys are a 400, not a find()."""
    dbs = _wire_mongo(mock_mongo_client)
    query = urllib.parse.quote(
        '{"_id": {"$oid": "64b8f7b7b1f1eaf5b2f0c001", "extra": 1}}'
    )
    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "oid" in payload["error"].lower()
    dbs["user_db"].__getitem__.return_value.find.assert_not_called()


def test_delete_documents_converts_valid_oid_to_objectid(client, mock_mongo_client) -> None:
    """DELETE uses the same $oid conversion so scoped deletes hit the intended _id."""
    dbs = _wire_mongo(mock_mongo_client)
    oid = ObjectId("64b8f7b7b1f1eaf5b2f0c001")
    delete_result = MagicMock()
    delete_result.deleted_count = 1
    dbs["user_db"].__getitem__.return_value.delete_many.return_value = delete_result

    query = urllib.parse.quote('{"_id": {"$oid": "64b8f7b7b1f1eaf5b2f0c001"}}')
    response = client.delete(
        f"/documents?database_name=test_db&collection_name=docs&query={query}"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted_count"] == 1
    dbs["user_db"].__getitem__.return_value.delete_many.assert_called_once_with(
        {"_id": oid}
    )
