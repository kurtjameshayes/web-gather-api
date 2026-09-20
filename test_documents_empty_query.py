"""Leftover GET /documents query-empty vs whitespace-only contracts."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
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
    user_db.__getitem__.return_value = MagicMock()
    return {"user_db": user_db}


def _cursor_with_docs(docs: list) -> MagicMock:
    cursor = MagicMock()
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def test_list_documents_empty_query_is_unscoped(client, mock_mongo_client) -> None:
    """query= (empty) is falsy, so find({}) runs — same as omitting query."""
    dbs = _wire_mongo(mock_mongo_client)
    cursor = _cursor_with_docs([])
    dbs["user_db"].__getitem__.return_value.find.return_value = cursor

    response = client.get(
        "/documents?database_name=test_db&collection_name=docs&query="
    )

    assert response.status_code == 200
    dbs["user_db"].__getitem__.return_value.find.assert_called_once_with({})


def test_list_documents_whitespace_query_is_invalid_json(client, mock_mongo_client) -> None:
    """Whitespace-only query is truthy and fails JSON parse before find()."""
    dbs = _wire_mongo(mock_mongo_client)

    response = client.get(
        f"/documents?database_name=test_db&collection_name=docs&query={quote(' ')}"
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.get_json()["error"]
    dbs["user_db"].__getitem__.return_value.find.assert_not_called()
