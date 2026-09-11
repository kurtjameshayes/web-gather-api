"""POST /create-embeddings privacy-compliance application-model preference."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import PRIVACY_COMPLIANCE_DB, WEB_GATHER_DB, core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


def _wire_mongo(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_db = MagicMock()
    privacy_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        if name == WEB_GATHER_DB:
            return wg_db
        if name == PRIVACY_COMPLIANCE_DB:
            return privacy_db
        return source_db

    mock_mongo.__getitem__.side_effect = _get_db
    source_db.__getitem__.return_value = MagicMock()
    privacy_db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
    return {"source_db": source_db, "privacy_db": privacy_db, "wg_db": wg_db}


class DummyModel:
    def encode(self, inputs, **kwargs: object) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in inputs])


def _source_row() -> dict:
    return {"_id": "row-1", "chunk_text": "Indexed policy text"}


def test_create_embeddings_privacy_compliance_prefers_application_model(
    client, mock_clients
) -> None:
    """Indexing privacy-compliance uses get_application_embedding_model before per-database lookup."""
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["source_db"].__getitem__.return_value.find.return_value = [_source_row()]
    dummy = DummyModel()

    with patch("core.get_application_embedding_model", return_value="app-model") as app_model, patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", return_value=dummy) as get_model:
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": PRIVACY_COMPLIANCE_DB,
                "index_collection_name": "policy_legal_embeddings",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["embedding_model"] == "app-model"
    app_model.assert_called_once()
    db_model.assert_not_called()
    get_model.assert_called_once_with("app-model")


def test_create_embeddings_privacy_compliance_falls_back_to_database_model(
    client, mock_clients
) -> None:
    """When the application model is unset, privacy-compliance falls back to the per-database record."""
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["source_db"].__getitem__.return_value.find.return_value = [_source_row()]

    with patch("core.get_application_embedding_model", return_value=None), patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", return_value=DummyModel()):
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": PRIVACY_COMPLIANCE_DB,
                "index_collection_name": "policy_legal_embeddings",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["embedding_model"] == "db-model"
    db_model.assert_called_once_with(PRIVACY_COMPLIANCE_DB)


def test_create_embeddings_other_database_skips_application_model(
    client, mock_clients
) -> None:
    """Non-privacy-compliance index targets never consult the application embedding model."""
    dbs = _wire_mongo(mock_clients["mongo"])
    dbs["source_db"].__getitem__.return_value.find.return_value = [_source_row()]

    with patch("core.get_application_embedding_model", return_value="app-model") as app_model, patch(
        "core.get_embedding_model_name", return_value="db-model"
    ), patch("core.get_model", return_value=DummyModel()) as get_model:
        response = client.post(
            "/create-embeddings",
            json={
                "source_database_name": "src",
                "source_collection_name": "docs",
                "index_database_name": "other_db",
                "index_collection_name": "chunks",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["embedding_model"] == "db-model"
    app_model.assert_not_called()
    get_model.assert_called_once_with("db-model")
