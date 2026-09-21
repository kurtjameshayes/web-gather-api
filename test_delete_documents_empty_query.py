"""Leftover DELETE /documents empty vs whitespace vs {} query wipe contracts."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from flask import Flask

from db import db_bp, init_db


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
    user_db = MagicMock()
    mock_mongo.__getitem__.return_value = user_db
    user_db.__getitem__.return_value = MagicMock()
    return {"user_db": user_db}


def test_delete_empty_query_string_is_unscoped_wipe(client, mock_mongo_client) -> None:
    """query= is falsy, so DELETE skips the parser and still calls delete_many({})."""
    dbs = _wire_mongo(mock_mongo_client)
    coll = dbs["user_db"].__getitem__.return_value
    coll.delete_many.return_value.deleted_count = 4

    response = client.delete(
        "/documents?database_name=test_db&collection_name=docs&query="
    )

    assert response.status_code == 200
    assert response.get_json()["deleted_count"] == 4
    coll.delete_many.assert_called_once_with({})


def test_delete_whitespace_query_is_invalid_json_before_delete(
    client, mock_mongo_client
) -> None:
    """Whitespace-only query is 400 Invalid JSON and must not delete."""
    dbs = _wire_mongo(mock_mongo_client)
    coll = dbs["user_db"].__getitem__.return_value

    response = client.delete(
        "/documents?database_name=test_db&collection_name=docs&query=" + quote("   ")
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.get_json()["error"]
    coll.delete_many.assert_not_called()


def test_delete_empty_object_query_is_unscoped_wipe_via_parser(
    client, mock_mongo_client
) -> None:
    """query={} is truthy JSON, parses to {}, and still wipes the collection."""
    dbs = _wire_mongo(mock_mongo_client)
    coll = dbs["user_db"].__getitem__.return_value
    coll.delete_many.return_value.deleted_count = 2

    response = client.delete(
        "/documents?database_name=test_db&collection_name=docs&query=" + quote("{}")
    )

    assert response.status_code == 200
    assert response.get_json()["deleted_count"] == 2
    coll.delete_many.assert_called_once_with({})
