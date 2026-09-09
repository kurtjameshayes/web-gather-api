"""POST /vector-search privacy-compliance model selection and result shaping."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from bson import ObjectId

from conftest import DummyModel, _wire_mongo_core
from core import PRIVACY_COMPLIANCE_DB, core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_vector_search_privacy_compliance_prefers_application_model(client, mock_clients) -> None:
    """privacy-compliance uses get_application_embedding_model before per-database lookup."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = []
    captured = {}

    def _get_model(name):
        captured["model_name"] = name
        return DummyModel()

    with patch("core.get_application_embedding_model", return_value="app-model") as app_model, patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", side_effect=_get_model), patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "embedding"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": PRIVACY_COMPLIANCE_DB,
                "collection": "policy_legal_embeddings",
                "index": "vector_index",
                "query": "opt out of sale",
                "path": "embedding",
            },
        )
    assert response.status_code == 200
    assert captured["model_name"] == "app-model"
    app_model.assert_called_once()
    db_model.assert_not_called()


def test_vector_search_privacy_compliance_falls_back_to_database_model(client, mock_clients) -> None:
    """When the application model is unset, privacy-compliance uses the per-database record."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = []
    captured = {}

    def _get_model(name):
        captured["model_name"] = name
        return DummyModel()

    with patch("core.get_application_embedding_model", return_value=None), patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as db_model, patch("core.get_model", side_effect=_get_model), patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "embedding"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": PRIVACY_COMPLIANCE_DB,
                "collection": "policy_legal_embeddings",
                "index": "vector_index",
                "query": "right to delete",
                "path": "embedding",
            },
        )
    assert response.status_code == 200
    assert captured["model_name"] == "db-model"
    db_model.assert_called_once_with(PRIVACY_COMPLIANCE_DB)


def test_vector_search_non_privacy_db_skips_application_model(client, mock_clients) -> None:
    """Non-privacy databases never consult the application embedding model."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = []

    with patch("core.get_application_embedding_model") as app_model, patch(
        "core.get_embedding_model_name", return_value="other-model"
    ), patch("core.get_model", return_value=DummyModel()), patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "embedding"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": "other_db",
                "collection": "chunks",
                "index": "idx",
                "query": "retention",
                "path": "embedding",
            },
        )
    assert response.status_code == 200
    app_model.assert_not_called()


def test_vector_search_stringifies_objectid_and_strips_vector_path(client, mock_clients) -> None:
    """Results stringify Mongo _id and drop the vector field named by path."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    oid = ObjectId("507f1f77bcf86cd799439011")
    mock_coll.aggregate.return_value = [
        {"_id": oid, "text": "match", "embedding": [0.1, 0.2], "score": 0.9},
    ]

    response = client.post(
        "/vector-search",
        json={
            "database": "test_db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
        },
    )
    assert response.status_code == 200
    result = response.get_json()["results"][0]
    assert result["_id"] == str(oid)
    assert "embedding" not in result
    assert result["text"] == "match"


def test_vector_search_aggregation_failure_is_json_500(client, mock_clients) -> None:
    """Atlas aggregate exceptions are mapped to JSON 500, not a generic Flask error page."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.side_effect = RuntimeError("index not found")

    response = client.post(
        "/vector-search",
        json={
            "database": "test_db",
            "collection": "chunks",
            "index": "missing_idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
        },
    )
    assert response.status_code == 500
    assert "Vector search failed" in response.get_json()["error"]
    assert "index not found" in response.get_json()["error"]
