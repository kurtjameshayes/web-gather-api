"""POST /write_to_collection does not lowercase mode (unlike /ingest).

APPEND/REPLACE are 400. Replace without update_id is 400 before Mongo.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def collection() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(collection: MagicMock):
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


@pytest.mark.parametrize("mode", ["APPEND", "REPLACE", "Overwrite", "append "])
def test_write_to_collection_mode_is_case_sensitive(client, collection, mode: str) -> None:
    """Unlike POST /ingest (which lowercases mode), uppercase/padded values are invalid."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": mode,
            "document": {"name": "doc1"},
            "update_id": "64b8f7b7b1f1eaf5b2f0c001",
        },
    )
    assert response.status_code == 400
    assert "mode" in response.get_json()["error"].lower()
    collection.insert_one.assert_not_called()
    collection.replace_one.assert_not_called()


def test_write_to_collection_replace_requires_update_id(client, collection) -> None:
    """mode=replace with omitted or empty update_id is 400 before ObjectId/Mongo."""
    omitted = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "document": {"name": "doc1"},
        },
    )
    assert omitted.status_code == 400
    assert "update_id" in omitted.get_json()["error"]

    empty = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "mode": "replace",
            "document": {"name": "doc1"},
            "update_id": "",
        },
    )
    assert empty.status_code == 400
    assert "update_id" in empty.get_json()["error"]
    collection.replace_one.assert_not_called()
    collection.insert_one.assert_not_called()
