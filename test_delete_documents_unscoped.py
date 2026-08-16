"""Regression tests for DELETE /documents when query is omitted.

Omitting query currently deletes every document in the collection. That
unscoped write path is high-risk and was not asserted on base.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

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
    wg_db.__getitem__.return_value = MagicMock()
    user_db.__getitem__.return_value = MagicMock()
    return {"wg_db": wg_db, "user_db": user_db}


def test_delete_documents_without_query_deletes_all(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    collection = dbs["user_db"].__getitem__.return_value
    delete_result = MagicMock()
    delete_result.deleted_count = 12
    collection.delete_many.return_value = delete_result

    response = client.delete(
        "/documents?database_name=test_db&collection_name=docs"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted_count"] == 12
    assert payload["database_name"] == "test_db"
    assert payload["collection_name"] == "docs"
    collection.delete_many.assert_called_once_with({})


def test_list_collections_requires_database_name(client, mock_mongo_client) -> None:
    response = client.get("/collections")
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]


def test_list_all_collections_requires_database_name(client, mock_mongo_client) -> None:
    response = client.get("/all-collections")
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]
