"""Regression tests for POST /write_to_collection replace success.

Base tests cover invalid ObjectId and not-found. Guard-only coverage in open
PRs does not assert the successful replace_one contract.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
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


def test_write_to_collection_replace_success(client, mock_mongo_client) -> None:
    dbs = _wire_mongo(mock_mongo_client)
    collection = dbs["user_db"].__getitem__.return_value
    replace_result = MagicMock()
    replace_result.matched_count = 1
    replace_result.modified_count = 1
    collection.replace_one.return_value = replace_result

    update_id = "64b8f7b7b1f1eaf5b2f0c001"
    document = {"name": "replaced", "status": "active"}
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "update_id": update_id,
            "document": document,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "replace"
    assert payload["update_id"] == update_id
    assert payload["matched_count"] == 1
    assert payload["modified_count"] == 1
    assert "replaced successfully" in payload["message"].lower()
    collection.replace_one.assert_called_once_with(
        {"_id": ObjectId(update_id)},
        document,
    )
    collection.insert_one.assert_not_called()


def test_write_to_collection_replace_reports_unmodified_match(client, mock_mongo_client) -> None:
    """A matching document with identical content still succeeds (modified_count=0)."""
    dbs = _wire_mongo(mock_mongo_client)
    collection = dbs["user_db"].__getitem__.return_value
    replace_result = MagicMock()
    replace_result.matched_count = 1
    replace_result.modified_count = 0
    collection.replace_one.return_value = replace_result

    update_id = "64b8f7b7b1f1eaf5b2f0c002"
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "update_id": update_id,
            "document": {"name": "same"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["matched_count"] == 1
    assert payload["modified_count"] == 0
    collection.replace_one.assert_called_once()
