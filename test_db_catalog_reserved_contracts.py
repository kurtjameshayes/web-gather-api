"""Catalog vs live Mongo reserved-database listing contracts."""
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


def test_list_collections_allows_reserved_database_catalog_lookup(
    client, mock_mongo_client
) -> None:
    """GET /collections queries the documents catalog and currently does not reject reserved names."""
    dbs = _wire_mongo(mock_mongo_client)
    catalog = dbs["wg_db"].__getitem__.return_value
    catalog.distinct.return_value = ["statutes"]

    response = client.get("/collections?database_name=admin")
    assert response.status_code == 200
    assert response.get_json()["collections"] == ["statutes"]
    catalog.distinct.assert_called_once_with(
        "collection_name", {"database_name": "admin"}
    )
    dbs["user_db"].list_collection_names.assert_not_called()


def test_list_all_collections_rejects_reserved_database(
    client, mock_mongo_client
) -> None:
    """GET /all-collections rejects reserved databases before enumerating live collections."""
    dbs = _wire_mongo(mock_mongo_client)

    response = client.get("/all-collections?database_name=ADMIN")
    assert response.status_code == 400
    assert "reserved" in response.get_json()["error"].lower()
    dbs["user_db"].list_collection_names.assert_not_called()


def test_list_databases_returns_reserved_catalog_names(
    client, mock_mongo_client
) -> None:
    """GET /databases returns distinct catalog names, including reserved ones if present."""
    dbs = _wire_mongo(mock_mongo_client)
    dbs["wg_db"].__getitem__.return_value.distinct.return_value = [
        "admin",
        "privacy-compliance",
    ]

    response = client.get("/databases")
    assert response.status_code == 200
    assert response.get_json()["databases"] == ["admin", "privacy-compliance"]
    mock_mongo_client.list_database_names.assert_not_called()


def test_list_all_databases_omits_reserved_names_case_insensitive(
    client, mock_mongo_client
) -> None:
    """GET /all-databases omits admin/config/local regardless of case."""
    mock_mongo_client.list_database_names.return_value = [
        "Admin",
        "CONFIG",
        "local",
        "privacy-compliance",
    ]

    response = client.get("/all-databases")
    assert response.status_code == 200
    assert response.get_json()["databases"] == ["privacy-compliance"]
