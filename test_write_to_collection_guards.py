"""Regression tests for write_to_collection input and reserved-DB guards."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import WEB_GATHER_DB, db_bp, init_db


@pytest.fixture
def mock_mongo_client() -> MagicMock:
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


@pytest.fixture
def client(mock_mongo_client: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    wg_db = MagicMock()
    user_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.return_value = MagicMock()
    user_db.__getitem__.return_value = MagicMock()
    return {"wg_db": wg_db, "user_db": user_db}


def test_write_to_collection_rejects_reserved_database(client, mock_mongo_client) -> None:
    """Writes must not target Mongo reserved DBs (admin/config/local)."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "admin",
            "collection_name": "users",
            "document": {"name": "x"},
        },
    )
    assert response.status_code == 400
    assert "reserved" in response.get_json()["error"].lower()
    mock_mongo_client.__getitem__.assert_not_called()


@pytest.mark.parametrize("mode", ["upsert", "delete", ""])
def test_write_to_collection_rejects_invalid_mode(client, mock_mongo_client, mode: str) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": mode,
            "document": {"name": "x"},
        },
    )
    assert response.status_code == 400
    assert "mode" in response.get_json()["error"].lower()


def test_write_to_collection_requires_document_object(client, mock_mongo_client) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "document": ["not", "an", "object"],
        },
    )
    assert response.status_code == 400
    assert "json object" in response.get_json()["error"].lower()


def test_write_to_collection_requires_document(client, mock_mongo_client) -> None:
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "append",
        },
    )
    assert response.status_code == 400
    assert "document is required" in response.get_json()["error"].lower()


def test_write_to_collection_rejects_reserved_before_insert(client, mock_mongo_client) -> None:
    """Reserved-DB rejection must happen before any collection write."""
    dbs = _wire_mongo(mock_mongo_client)
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "local",
            "collection_name": "startup_log",
            "document": {"ok": True},
        },
    )
    assert response.status_code == 400
    dbs["user_db"].__getitem__.return_value.insert_one.assert_not_called()
