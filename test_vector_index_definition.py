"""POST /create-vector-index definition assembly.

Whitespace database/collection names, whitespace-only index names, omitted
filter_fields, an explicit empty filter list, and a non-list filter_fields
value are covered elsewhere and are not retested.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from flask import Flask

from util import init_util, util_bp


def _client(mongo: MagicMock):
    init_util(mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), mongo


def _target_db() -> MagicMock:
    database = MagicMock()
    database.command.side_effect = [
        {"ok": 1},
        {"ok": 1},
    ]
    return database


def test_create_vector_index_forces_cosine_and_normalizes_filter_paths() -> None:
    """Vector fields are rewritten to cosine. Other fields and the stored record stay as-is.

    Blank filter paths are dropped. A numeric path is stringified. A padded index
    name is kept, including its spaces.
    """
    target = _target_db()
    mongo = MagicMock()
    mongo.__getitem__.return_value = target
    client, _mongo = _client(mongo)
    vector_field = {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 384,
        "similarity": "dotProduct",
    }
    record = {
        "database_name": "test_db",
        "fields": [
            vector_field,
            {"type": "filter", "path": "keep_me", "similarity": "euclidean"},
            "raw-field",
        ],
    }

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "index_name": " idx ",
                "filter_fields": [
                    "  ",
                    " document_id ",
                    {"path": "  "},
                    {"path": 12},
                    {"path": 0},
                    {"path": True},
                    {"missing": "x"},
                ],
            },
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["index_name"] == " idx "
    assert body["filter_fields_added"] == 3
    assert body["definition"]["fields"] == [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "keep_me", "similarity": "euclidean"},
        "raw-field",
        {"type": "filter", "path": "document_id"},
        {"type": "filter", "path": "12"},
        {"type": "filter", "path": "True"},
    ]
    assert vector_field["similarity"] == "dotProduct"
    drop_call, create_call = target.command.call_args_list
    assert drop_call.args[0] == {"dropSearchIndex": "coll", "name": " idx "}
    created = create_call.args[0]
    assert created["createSearchIndexes"] == "coll"
    assert created["indexes"][0]["name"] == " idx "
    assert created["indexes"][0]["type"] == "vectorSearch"
    assert created["indexes"][0]["definition"] == body["definition"]


def test_create_vector_index_empty_index_name_defaults_to_vector_index() -> None:
    target = _target_db()
    mongo = MagicMock()
    mongo.__getitem__.return_value = target
    client, _mongo = _client(mongo)
    record = {"fields": [{"type": "vector", "path": "embedding", "similarity": "cosine"}]}

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "index_name": "",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["index_name"] == "vector_index"
    assert target.command.call_args_list[1].args[0]["indexes"][0]["name"] == "vector_index"


def test_create_vector_index_non_string_index_name_is_400_before_lookup() -> None:
    mongo = MagicMock()
    client, _mongo = _client(mongo)

    with patch("util.get_embedding_model_record") as lookup:
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "index_name": 12,
            },
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "index_name must be a non-empty string when provided"
    lookup.assert_not_called()
    mongo.__getitem__.assert_not_called()


def test_create_vector_index_empty_fields_is_400_before_command() -> None:
    target = _target_db()
    mongo = MagicMock()
    mongo.__getitem__.return_value = target
    client, _mongo = _client(mongo)

    with patch("util.get_embedding_model_record", return_value={"fields": []}):
        response = client.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    assert response.status_code == 400
    assert "no valid 'fields' array" in response.get_json()["error"]
    target.command.assert_not_called()


def test_create_vector_index_not_ok_and_command_error_are_500() -> None:
    target = MagicMock()
    target.command.side_effect = [
        Exception("missing index"),
        {"ok": 0, "errmsg": "nope"},
    ]
    mongo = MagicMock()
    mongo.__getitem__.return_value = target
    client, _mongo = _client(mongo)
    record = {"fields": [{"type": "vector", "path": "embedding"}]}

    with patch("util.get_embedding_model_record", return_value=record):
        not_ok = client.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    assert not_ok.status_code == 500
    assert "createSearchIndexes failed:" in not_ok.get_json()["error"]
    assert "nope" in not_ok.get_json()["error"]

    target.command.side_effect = [
        {"ok": 1},
        RuntimeError("atlas down"),
    ]
    with patch("util.get_embedding_model_record", return_value=record):
        exploded = client.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    assert exploded.status_code == 500
    assert exploded.get_json()["error"] == "Failed to create vector index: atlas down"
