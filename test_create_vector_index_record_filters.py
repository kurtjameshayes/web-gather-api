"""POST /create-vector-index filter_fields merge/override contract.

Request `filter_fields` replace embedding_model.filter_fields rather than
merging. Blank strings and blank-path objects are skipped. Non-list
filter_fields is 400 before any Atlas command. Vector similarity is forced
to cosine even when the stored model says otherwise.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from flask import Flask

from util import init_util, util_bp


def _client(record: dict):
    target_db = MagicMock()
    target_db.command.side_effect = [
        {"ok": 1},  # dropSearchIndex
        {"ok": 1},  # createSearchIndexes
    ]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), target_db, record


def test_create_vector_index_uses_record_filter_fields_when_omitted() -> None:
    """Omitting filter_fields uses embedding_model.filter_fields (skipping blanks)."""
    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "euclidean",
            }
        ],
        "filter_fields": [
            "document_id",
            "  ",
            {"path": "jurisdiction"},
            {"path": "   "},
            "category",
        ],
    }
    client, target_db, _ = _client(record)

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filter_fields_added"] == 3
    definition = payload["definition"]
    vector_fields = [f for f in definition["fields"] if isinstance(f, dict) and f.get("type") == "vector"]
    filter_fields = [f for f in definition["fields"] if isinstance(f, dict) and f.get("type") == "filter"]
    assert vector_fields[0]["similarity"] == "cosine"
    assert [f["path"] for f in filter_fields] == ["document_id", "jurisdiction", "category"]
    assert target_db.command.call_count == 2


def test_create_vector_index_request_filter_fields_replace_record() -> None:
    """Body filter_fields replace (do not AND-merge) the stored model filters."""
    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"}
        ],
        "filter_fields": ["document_id", "jurisdiction"],
    }
    client, _target_db, _ = _client(record)

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "filter_fields": ["policy_id"],
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filter_fields_added"] == 1
    filter_fields = [
        f for f in payload["definition"]["fields"] if isinstance(f, dict) and f.get("type") == "filter"
    ]
    assert [f["path"] for f in filter_fields] == ["policy_id"]


def test_create_vector_index_non_list_filter_fields_is_400() -> None:
    """Non-array filter_fields is 400 and must not call Atlas commands."""
    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"}
        ],
    }
    client, target_db, _ = _client(record)

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "filter_fields": "document_id",
            },
        )

    assert response.status_code == 400
    assert "filter_fields" in response.get_json()["error"]
    target_db.command.assert_not_called()
