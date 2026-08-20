"""POST /create-vector-index drop-then-create sequencing.

Error mapping and filter_fields merge are covered in other PRs. These tests pin
the Atlas mutation order: dropSearchIndex is attempted first, drop failures are
non-fatal, and createSearchIndexes still runs with the default index name.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from flask import Flask

from util import init_util, util_bp

VECTOR_FIELD = {
    "type": "vector",
    "path": "embedding",
    "numDimensions": 384,
    "similarity": "cosine",
}


def _client_with_db(target_db: MagicMock):
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), mock_mongo


def _record() -> dict:
    return {
        "database_name": "privacy-compliance",
        "model_name": "all-MiniLM-L6-v2",
        "fields": [VECTOR_FIELD],
    }


def test_create_vector_index_drops_existing_index_then_creates() -> None:
    target_db = MagicMock()
    target_db.command.side_effect = [{"ok": 1}, {"ok": 1}]
    client, _ = _client_with_db(target_db)

    with patch("util.get_embedding_model_record", return_value=_record()):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "statute_embeddings",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["index_name"] == "vector_index"
    assert payload["filter_fields_added"] == 0
    assert payload["definition"]["fields"] == [VECTOR_FIELD]
    assert target_db.command.call_args_list == [
        call({"dropSearchIndex": "statute_embeddings", "name": "vector_index"}),
        call(
            {
                "createSearchIndexes": "statute_embeddings",
                "indexes": [
                    {
                        "name": "vector_index",
                        "type": "vectorSearch",
                        "definition": {"fields": [VECTOR_FIELD]},
                    }
                ],
            }
        ),
    ]


def test_create_vector_index_continues_when_drop_fails_because_index_missing() -> None:
    target_db = MagicMock()
    target_db.command.side_effect = [Exception("index not found"), {"ok": 1}]
    client, _ = _client_with_db(target_db)

    with patch("util.get_embedding_model_record", return_value=_record()):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "policy_embeddings",
                "index_name": "custom_index",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["index_name"] == "custom_index"
    drop_call, create_call = target_db.command.call_args_list
    assert drop_call == call(
        {"dropSearchIndex": "policy_embeddings", "name": "custom_index"}
    )
    assert create_call.args[0]["createSearchIndexes"] == "policy_embeddings"
    assert create_call.args[0]["indexes"][0]["name"] == "custom_index"
