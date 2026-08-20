"""Reserved-database listing and access guards.

Base tests only assert happy-path /all-databases and /all-collections payloads.
These pin the security contract that system databases are not listed or queried.
"""
from __future__ import annotations

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
def mock_mongo() -> MagicMock:
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


def test_all_databases_excludes_reserved_system_names(client, mock_mongo) -> None:
    """GET /all-databases must omit admin/config/local regardless of casing."""
    mock_mongo.list_database_names.return_value = [
        "privacy-compliance",
        "admin",
        "CONFIG",
        "Local",
        "web-gather",
    ]

    response = client.get("/all-databases")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["databases"] == ["privacy-compliance", "web-gather"]


def test_all_collections_rejects_reserved_database(client, mock_mongo) -> None:
    """GET /all-collections must not enumerate collections in reserved DBs."""
    response = client.get("/all-collections?database_name=Admin")

    assert response.status_code == 400
    payload = response.get_json()
    assert "reserved" in payload["error"].lower()
    mock_mongo.__getitem__.assert_not_called()


def test_all_collections_rejects_local_database(client, mock_mongo) -> None:
    response = client.get("/all-collections?database_name=local")

    assert response.status_code == 400
    assert "reserved" in response.get_json()["error"].lower()
    mock_mongo.__getitem__.assert_not_called()


def test_all_collections_requires_database_name(client, mock_mongo) -> None:
    response = client.get("/all-collections")

    assert response.status_code == 400
    assert "database_name is required" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()


def test_collections_requires_database_name(client, mock_mongo) -> None:
    """GET /collections lists uploaded collections and still requires database_name."""
    response = client.get("/collections")

    assert response.status_code == 400
    assert "database_name is required" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()


def test_all_collections_success_does_not_filter_user_databases(client, mock_mongo) -> None:
    user_db = MagicMock()
    user_db.list_collection_names.return_value = ["policies", "chunks"]
    mock_mongo.__getitem__.return_value = user_db

    response = client.get("/all-collections?database_name=privacy-compliance")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "privacy-compliance"
    assert payload["collections"] == ["policies", "chunks"]
    mock_mongo.__getitem__.assert_called_with("privacy-compliance")
    # Guard against accidentally querying the metadata DB instead of the target.
    assert WEB_GATHER_DB != "privacy-compliance"
