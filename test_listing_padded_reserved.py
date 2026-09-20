"""Leftover reserved-name listing contracts for padded admin/config/local."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

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


def test_all_databases_keeps_padded_reserved_names(client, mock_mongo_client) -> None:
    """Exact admin/config/local (any case) are omitted; padded names are listed."""
    mock_mongo_client.list_database_names.return_value = [
        "user_db",
        "admin",
        "ADMIN",
        " Admin",
        "admin ",
        "config",
        "Config",
        "local",
    ]

    response = client.get("/all-databases")

    assert response.status_code == 200
    databases = response.get_json()["databases"]
    assert databases == ["user_db", " Admin", "admin "]
    mock_mongo_client.list_database_names.assert_called_once_with()


def test_collections_padded_reserved_uses_catalog(client, mock_mongo_client) -> None:
    """GET /collections has no reserved-name guard; ' admin' is a catalog filter."""
    catalog = MagicMock()
    catalog.distinct.return_value = ["policies"]
    wg_db = MagicMock()
    wg_db.__getitem__.return_value = catalog
    mock_mongo_client.__getitem__.return_value = wg_db

    response = client.get("/collections?database_name=%20admin")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == " admin"
    assert payload["collections"] == ["policies"]
    mock_mongo_client.__getitem__.assert_called_with(WEB_GATHER_DB)
    wg_db.__getitem__.assert_called_with(DOCUMENTS_COLLECTION)
    catalog.distinct.assert_called_once_with(
        "collection_name", {"database_name": " admin"}
    )
    mock_mongo_client.__getitem__.return_value.list_collection_names.assert_not_called()
