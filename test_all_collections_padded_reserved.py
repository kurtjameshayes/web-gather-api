"""GET /all-collections reserved-name matching is exact after lowercasing.

Case-insensitive `ADMIN` is rejected. A padded value such as ` admin` is not in
the reserved set and currently proceeds to live `list_collection_names`.
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


def test_list_all_collections_padded_reserved_name_is_not_rejected(
    client, mock_mongo_client
) -> None:
    """Whitespace around a reserved name currently bypasses the reserved check."""
    dbs = _wire_mongo(mock_mongo_client)
    dbs["user_db"].list_collection_names.return_value = ["system.users"]

    response = client.get("/all-collections?database_name=%20admin")

    assert response.status_code == 200
    assert response.get_json()["database_name"] == " admin"
    assert response.get_json()["collections"] == ["system.users"]
    mock_mongo_client.__getitem__.assert_called_with(" admin")
    dbs["user_db"].list_collection_names.assert_called_once_with()
