"""GET /count-documents does not strip or reserved-check padded names.

#214 pins unguarded `admin`/`ADMIN`. A leading space means
`database_name.lower()` is ` admin`, which is not in RESERVED_DB_NAMES, so
the handler forwards to that Mongo database. Distinct from #238
GET /all-collections padded reserved (list, not count).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def mock_mongo():
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


@pytest.fixture
def client(mock_mongo):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_count_documents_padded_reserved_name_is_forwarded(client, mock_mongo) -> None:
    """database_name=' admin' is counted on that exact name, not rejected."""
    padded_db = MagicMock()
    coll = MagicMock()
    coll.count_documents.return_value = 3

    def _get_db(name: str):
        if name == " admin":
            return padded_db
        return MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    padded_db.__getitem__.return_value = coll

    response = client.get(
        "/count-documents?database_name=%20admin&collection_name=system.users"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["database_name"] == " admin"
    assert data["collection_name"] == "system.users"
    assert data["count"] == 3
    mock_mongo.__getitem__.assert_called_with(" admin")
    coll.count_documents.assert_called_once_with({})
