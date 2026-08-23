"""Regression tests pinning GET /count-documents validation gaps.

Unlike GET/DELETE /documents and POST /write_to_collection, count-documents
currently has no reserved-database guard and no collection-name character
check. Pin that contract so a later hardening change is intentional.
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


def test_count_documents_allows_reserved_database(client, mock_mongo) -> None:
    """GET /count-documents currently does not reject admin/config/local."""
    collection = MagicMock()
    collection.count_documents.return_value = 3
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection

    response = client.get(
        "/count-documents?database_name=admin&collection_name=system.users"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "admin"
    assert payload["collection_name"] == "system.users"
    assert payload["count"] == 3
    mock_mongo.__getitem__.assert_called_with("admin")
    collection.count_documents.assert_called_once_with({})


def test_count_documents_allows_invalid_collection_characters(
    client, mock_mongo
) -> None:
    """Spaces in names are accepted here even though /documents rejects them."""
    collection = MagicMock()
    collection.count_documents.return_value = 0
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection

    response = client.get(
        "/count-documents?database_name=bad name&collection_name=also bad"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "bad name"
    assert payload["collection_name"] == "also bad"
    assert payload["count"] == 0
    mock_mongo.__getitem__.assert_called_with("bad name")
