"""Regression tests for GET /search privacy-compliance embedding-model selection.

When the index database is privacy-compliance, search prefers the process-wide
application embedding model and only falls back to the per-database record.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import PRIVACY_COMPLIANCE_DB, WEB_GATHER_DB, core_bp, init_core
from db import get_application_embedding_model, set_application_embedding_model


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0]])


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients) -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


@pytest.fixture(autouse=True)
def reset_application_embedding_model():
    previous = get_application_embedding_model()
    set_application_embedding_model(None)
    yield
    set_application_embedding_model(previous)


def _wire_search_dbs(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    wg_db = MagicMock(name="wg_db")
    privacy_db = MagicMock(name="privacy_db")
    chunk_coll = MagicMock(name="chunk_coll")
    docs_coll = MagicMock(name="docs_coll")
    wg_db.__getitem__.return_value = docs_coll
    privacy_db.__getitem__.return_value = chunk_coll

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        return privacy_db

    mock_mongo.__getitem__.side_effect = _get_db
    docs_coll.find_one.return_value = {
        "document_id": "doc-1",
        "database_name": "src",
        "collection_name": "docs",
        "index_database_name": PRIVACY_COMPLIANCE_DB,
        "chunk_collection": "policy_chunks",
    }
    chunk_coll.find.return_value = [
        {"text": "privacy chunk", "embedding": [1.0, 0.0]},
    ]
    return {"docs_coll": docs_coll, "chunk_coll": chunk_coll}


def test_search_privacy_compliance_prefers_application_embedding_model(
    client, mock_clients
) -> None:
    """Application default wins over the per-database embedding_model record."""
    _wire_search_dbs(mock_clients["mongo"])
    set_application_embedding_model("app-default-model")
    per_db_lookup = MagicMock(return_value="per-db-model")

    with patch("core.get_embedding_model_name", per_db_lookup), patch(
        "core.get_model", return_value=DummyModel()
    ) as mock_get_model:
        response = client.get("/search?document_id=doc-1&query=retention")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["results"][0]["text"] == "privacy chunk"
    mock_get_model.assert_called_once_with("app-default-model")
    per_db_lookup.assert_not_called()


def test_search_privacy_compliance_falls_back_to_database_model(
    client, mock_clients
) -> None:
    """When no application default is set, search uses get_embedding_model_name."""
    _wire_search_dbs(mock_clients["mongo"])
    set_application_embedding_model(None)

    with patch("core.get_embedding_model_name", return_value="per-db-model") as per_db, patch(
        "core.get_model", return_value=DummyModel()
    ) as mock_get_model:
        response = client.get("/search?document_id=doc-1&query=retention")

    assert response.status_code == 200
    per_db.assert_called_once_with(PRIVACY_COMPLIANCE_DB)
    mock_get_model.assert_called_once_with("per-db-model")
