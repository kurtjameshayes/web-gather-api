"""POST /write_to_collection rejects invalid database/collection names.

#204 covers reserved databases and invalid modes. GET/DELETE /documents already
reject dotted and spaced names (#235 / base). Writes currently share the same
validate_collection_name guard; pin it so a missing check cannot insert into
dotted or whitespace names.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def mock_mongo() -> MagicMock:
    mock = MagicMock()
    init_db(mock)
    return mock


@pytest.fixture
def client(mock_mongo: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_write_to_collection_rejects_dotted_collection_name(client, mock_mongo) -> None:
    """Dotted collection names must 400 before insert_one."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "my.collection",
            "document": {"name": "x"},
        },
    )

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "letters, numbers" in error or "invalid" in error.lower()
    mock_mongo.__getitem__.assert_not_called()


def test_write_to_collection_rejects_spaced_database_name(client, mock_mongo) -> None:
    """Whitespace in database_name must 400 before insert_one."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test db",
            "collection_name": "docs",
            "document": {"name": "x"},
        },
    )

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "letters, numbers" in error or "invalid" in error.lower()
    mock_mongo.__getitem__.assert_not_called()


def test_write_to_collection_rejects_dollar_collection_name(client, mock_mongo) -> None:
    """Names with '$' fail validate_collection_name before any Mongo write."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "$cmd",
            "document": {"ok": True},
        },
    )

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "letters, numbers" in error or "invalid" in error.lower()
    mock_mongo.__getitem__.assert_not_called()
