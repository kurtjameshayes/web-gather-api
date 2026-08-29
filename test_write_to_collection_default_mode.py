"""Regression tests for POST /write_to_collection default mode and empty document."""
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


def test_write_to_collection_omitted_mode_is_append(client, collection) -> None:
    """Omitting mode defaults to append and calls insert_one."""
    collection.insert_one.return_value.inserted_id = "abc123"

    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "document": {"name": "doc1"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "append"
    assert payload["inserted_id"] == "abc123"
    collection.insert_one.assert_called_once_with({"name": "doc1"})
    collection.replace_one.assert_not_called()


def test_write_to_collection_empty_object_document_is_required(client, collection) -> None:
    """Empty dict {} is falsy and currently treated as a missing document (400)."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "document": {},
        },
    )

    assert response.status_code == 400
    assert "document is required" in response.get_json()["error"]
    collection.insert_one.assert_not_called()
    collection.replace_one.assert_not_called()


def test_write_to_collection_empty_list_document_is_required(client, collection) -> None:
    """Empty list is also falsy, so the error is 'required' rather than 'must be a JSON object'."""
    response = client.post(
        "/write_to_collection",
        json={
            "database_name": "test_db",
            "collection_name": "docs",
            "document": [],
        },
    )

    assert response.status_code == 400
    assert "document is required" in response.get_json()["error"]
    collection.insert_one.assert_not_called()
