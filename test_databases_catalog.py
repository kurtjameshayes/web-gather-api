"""Regression tests for catalog vs live Mongo listing endpoints."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import DOCUMENTS_COLLECTION, WEB_GATHER_DB, db_bp, init_db


@pytest.fixture
def mock_mongo() -> MagicMock:
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


@pytest.fixture
def app(mock_mongo):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _wire_catalog(mock_mongo: MagicMock) -> Dict[str, Any]:
    wg_db = MagicMock()
    docs = MagicMock()
    user_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else user_db

    def _get_coll(name: str) -> MagicMock:
        return docs if name == DOCUMENTS_COLLECTION else MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    wg_db.__getitem__.side_effect = _get_coll
    user_db.list_collection_names.return_value = ["live-coll"]
    return {"wg_db": wg_db, "docs": docs, "user_db": user_db}


def test_list_databases_uses_documents_catalog_not_live_mongo(client, mock_mongo) -> None:
    """GET /databases must read uploaded-document metadata, not list_database_names()."""
    wired = _wire_catalog(mock_mongo)
    wired["docs"].distinct.return_value = ["privacy-compliance", "statutes"]

    response = client.get("/databases")
    assert response.status_code == 200
    assert response.get_json()["databases"] == ["privacy-compliance", "statutes"]
    wired["docs"].distinct.assert_called_once_with("database_name")
    mock_mongo.list_database_names.assert_not_called()


def test_list_collections_uses_documents_catalog_filter(client, mock_mongo) -> None:
    """GET /collections must distinct collection_name for the requested database."""
    wired = _wire_catalog(mock_mongo)
    wired["docs"].distinct.return_value = ["policies", "statute_chunks"]

    response = client.get("/collections?database_name=privacy-compliance")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "privacy-compliance"
    assert payload["collections"] == ["policies", "statute_chunks"]
    wired["docs"].distinct.assert_called_once_with(
        "collection_name", {"database_name": "privacy-compliance"}
    )
    wired["user_db"].list_collection_names.assert_not_called()


def test_list_collections_does_not_apply_reserved_db_guard(client, mock_mongo) -> None:
    """GET /collections currently allows reserved names and still queries the catalog."""
    wired = _wire_catalog(mock_mongo)
    wired["docs"].distinct.return_value = []

    response = client.get("/collections?database_name=admin")
    assert response.status_code == 200
    wired["docs"].distinct.assert_called_once_with(
        "collection_name", {"database_name": "admin"}
    )
