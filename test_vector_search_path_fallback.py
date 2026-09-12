"""POST /vector-search omitted-path and numCandidates contracts.

When `path` is omitted, the first embedding_model.fields[].path is used even if
that field is not type=vector. numCandidates is max(limit*10, 100). Nearby
coverage pins limit clamps, filters, and provided-path pipelines, not this
fallback.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _first_stage(mock_mongo: MagicMock) -> dict:
    coll = mock_mongo.__getitem__.return_value.__getitem__.return_value
    coll.aggregate.return_value = []
    return coll


def test_vector_search_omitted_path_uses_first_field_even_if_not_vector(
    client, mock_clients
) -> None:
    coll = _first_stage(mock_clients)
    record = {
        "fields": [
            {"type": "filter", "path": "jurisdiction"},
            {"type": "vector", "path": "embedding"},
        ]
    }

    with patch("core.get_embedding_model_record", return_value=record):
        response = client.post(
            "/vector-search",
            json={
                "database": "privacy-compliance",
                "collection": "statute_sub_embeddings",
                "index": "vector_index",
                "query_vector": [0.1, 0.2],
                "limit": 3,
            },
        )

    assert response.status_code == 200
    pipeline = coll.aggregate.call_args[0][0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["path"] == "jurisdiction"
    assert stage["limit"] == 3
    assert stage["numCandidates"] == 100


def test_vector_search_num_candidates_scales_after_floor(client, mock_clients) -> None:
    coll = _first_stage(mock_clients)

    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1.0],
            "path": "embedding",
            "limit": 15,
        },
    )

    assert response.status_code == 200
    stage = coll.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert stage["limit"] == 15
    assert stage["numCandidates"] == 150


def test_vector_search_missing_path_and_empty_fields_is_400(client, mock_clients) -> None:
    coll = _first_stage(mock_clients)

    with patch("core.get_embedding_model_record", return_value={"fields": []}):
        response = client.post(
            "/vector-search",
            json={
                "database": "db",
                "collection": "chunks",
                "index": "idx",
                "query_vector": [1.0],
            },
        )

    assert response.status_code == 400
    assert "path" in response.get_json()["error"].lower()
    coll.aggregate.assert_not_called()


def test_vector_search_first_field_without_path_is_400(client, mock_clients) -> None:
    coll = _first_stage(mock_clients)

    with patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"type": "vector"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": "db",
                "collection": "chunks",
                "index": "idx",
                "query_vector": [1.0],
            },
        )

    assert response.status_code == 400
    assert "path" in response.get_json()["error"].lower()
    coll.aggregate.assert_not_called()
