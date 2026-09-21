"""Leftover GET /collections and /all-collections invalid-name forwarding."""
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


def test_all_collections_forwards_dotted_database_name(client, mock_mongo_client) -> None:
    """GET /all-collections does not run validate_collection_name (unlike /documents)."""
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].list_collection_names.return_value = ["coll_a"]

    response = client.get("/all-collections?database_name=test.db")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "test.db"
    assert payload["collections"] == ["coll_a"]
    mock_mongo_client.__getitem__.assert_any_call("test.db")
    dbs["user_db"].list_collection_names.assert_called_once_with()


def test_all_collections_forwards_dollar_database_name(client, mock_mongo_client) -> None:
    """Names like $cmd are not reserved-rejected and reach list_collection_names."""
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].list_collection_names.return_value = []

    response = client.get("/all-collections?database_name=$cmd")

    assert response.status_code == 200
    mock_mongo_client.__getitem__.assert_any_call("$cmd")
    dbs["user_db"].list_collection_names.assert_called_once_with()


def test_collections_catalog_lookup_uses_exact_invalid_name(
    client, mock_mongo_client
) -> None:
    """GET /collections has no name/reserved guards; dotted names are catalog filters."""
    dbs = _wire_mongo(mock_mongo_client)
    catalog = dbs["wg_db"].__getitem__.return_value
    catalog.distinct.return_value = ["ingested"]

    response = client.get("/collections?database_name=test.db")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "test.db"
    assert payload["collections"] == ["ingested"]
    catalog.distinct.assert_called_once_with(
        "collection_name", {"database_name": "test.db"}
    )
    dbs["user_db"].list_collection_names.assert_not_called()
