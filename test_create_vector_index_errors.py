"""Regression tests for /create-vector-index failure and definition guards.

Base and open filter-field PRs cover happy-path filter merge. These tests pin
blank index names, empty model fields, Atlas command failures, and drop-then-
create so a broken index definition cannot silently ship.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from flask import Flask

from util import init_util, util_bp


def _client(mock_mongo: MagicMock | None = None):
    init_util(mock_mongo or MagicMock())
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def test_create_vector_index_rejects_blank_index_name() -> None:
    client = _client()
    response = client.post(
        "/create-vector-index",
        json={
            "database_name": "privacy-compliance",
            "collection_name": "policy_embeddings",
            "index_name": "   ",
        },
    )
    assert response.status_code == 400
    assert "index_name" in response.get_json()["error"]


def test_create_vector_index_rejects_empty_model_fields() -> None:
    record = {"database_name": "privacy-compliance", "model_name": "m", "fields": []}
    with patch("util.get_embedding_model_record", return_value=record):
        client = _client()
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "policy_embeddings",
            },
        )
    assert response.status_code == 400
    assert "fields" in response.get_json()["error"]


def test_create_vector_index_returns_500_when_create_not_ok() -> None:
    target_db = MagicMock()
    target_db.command.side_effect = [
        {"ok": 1},  # drop existing index
        {"ok": 0, "errmsg": "atlas rejected definition"},
    ]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    record = {
        "database_name": "privacy-compliance",
        "model_name": "m",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "euclidean",
            }
        ],
    }
    with patch("util.get_embedding_model_record", return_value=record):
        client = _client(mock_mongo)
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "policy_embeddings",
                "index_name": "policy_vectors",
            },
        )
    assert response.status_code == 500
    assert "createSearchIndexes failed" in response.get_json()["error"]
    drop_cmd, create_cmd = target_db.command.call_args_list
    assert drop_cmd.args[0] == {
        "dropSearchIndex": "policy_embeddings",
        "name": "policy_vectors",
    }
    created = create_cmd.args[0]
    assert created["createSearchIndexes"] == "policy_embeddings"
    assert created["indexes"][0]["type"] == "vectorSearch"
    assert created["indexes"][0]["definition"]["fields"][0]["similarity"] == "cosine"


def test_create_vector_index_returns_500_when_create_raises() -> None:
    target_db = MagicMock()
    target_db.command.side_effect = [
        Exception("index missing"),
        RuntimeError("network to atlas failed"),
    ]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    record = {
        "database_name": "privacy-compliance",
        "model_name": "m",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine",
            }
        ],
    }
    with patch("util.get_embedding_model_record", return_value=record):
        client = _client(mock_mongo)
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "policy_embeddings",
            },
        )
    assert response.status_code == 500
    assert "Failed to create vector index" in response.get_json()["error"]
    assert "network to atlas failed" in response.get_json()["error"]
