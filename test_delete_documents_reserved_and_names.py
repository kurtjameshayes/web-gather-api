"""DELETE /documents reserved-DB and collection-name guards.

GET /documents already rejects reserved names and invalid characters on base.
DELETE currently has only success/query/unscoped coverage; pin the same
permission and name-validation contracts so a missing guard cannot wipe
admin/config/local or accept dotted names.
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


@pytest.mark.parametrize("reserved_name", ["admin", "ADMIN", "Config", "LOCAL"])
def test_delete_documents_rejects_reserved_database(
    client, mock_mongo: MagicMock, reserved_name: str
) -> None:
    """Reserved names are matched case-insensitively before any delete_many."""
    response = client.delete(
        f"/documents?database_name={reserved_name}&collection_name=users"
        "&query=%7B%22status%22%3A%22inactive%22%7D"
    )
    assert response.status_code == 400
    assert "reserved" in response.get_json()["error"].lower()
    mock_mongo.__getitem__.assert_not_called()


def test_delete_documents_rejects_invalid_collection_name(
    client, mock_mongo: MagicMock
) -> None:
    response = client.delete(
        "/documents?database_name=test_db&collection_name=my.collection"
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "letters, numbers" in error or "invalid" in error.lower()
    mock_mongo.__getitem__.assert_not_called()


def test_delete_documents_rejects_invalid_database_name(
    client, mock_mongo: MagicMock
) -> None:
    response = client.delete(
        "/documents?database_name=test db&collection_name=docs"
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "letters, numbers" in error or "invalid" in error.lower()
    mock_mongo.__getitem__.assert_not_called()


def test_list_documents_rejects_reserved_database_case_insensitive(client) -> None:
    """GET /documents uses the same .lower() reserved-name check as DELETE."""
    response = client.get("/documents?database_name=ADMIN&collection_name=users")
    assert response.status_code == 400
    assert "reserved" in response.get_json()["error"].lower()
