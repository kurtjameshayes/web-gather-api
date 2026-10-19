"""POST/GET /embedding-models name, similarity, and body validation.

Bool, float, and string numDimensions are covered elsewhere and are not retested.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from db import EMBEDDING_MODEL_COLLECTION, WEB_GATHER_DB
from util import init_util, util_bp


def _client(mongo: MagicMock):
    init_util(mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def _collection(mongo: MagicMock) -> MagicMock:
    collection = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = collection
    return collection


def _vector_field(**overrides):
    field = {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 384,
        "similarity": "cosine",
    }
    field.update(overrides)
    return field


def test_post_embedding_model_stores_padded_names_unstripped() -> None:
    """Whitespace around names is kept. Later lookups are exact, so the padding is the key."""
    mongo = MagicMock()
    collection = _collection(mongo)
    client = _client(mongo)
    payload = {
        "database_name": "  privacy-compliance  ",
        "model_name": "  mini  ",
        "fields": [_vector_field()],
    }

    with patch("util.utc_now", return_value="2026-01-01T00:00:00Z"):
        response = client.post("/embedding-models", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert body["database_name"] == "  privacy-compliance  "
    assert body["model_name"] == "  mini  "
    assert body["updated_at"] == "2026-01-01T00:00:00Z"
    assert body["fields"][0]["similarity"] == "cosine"
    collection.update_one.assert_called_once_with(
        {"database_name": "  privacy-compliance  "},
        {"$set": body},
        upsert=True,
    )
    mongo.__getitem__.assert_called_with(WEB_GATHER_DB)
    mongo.__getitem__.return_value.__getitem__.assert_called_with(EMBEDDING_MODEL_COLLECTION)


@pytest.mark.parametrize(
    ("payload", "error_snippet"),
    [
        (
            {"database_name": "   ", "model_name": "mini", "fields": [_vector_field()]},
            "database_name is required",
        ),
        (
            {"database_name": 12, "model_name": "mini", "fields": [_vector_field()]},
            "database_name is required",
        ),
        (
            {"database_name": "db", "model_name": "   ", "fields": [_vector_field()]},
            "model_name is required",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": []},
            "fields is required",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": {"path": "embedding"}},
            "fields is required",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": ["embedding"]},
            "fields[0] must be an object",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": [_vector_field(type="Vector")]},
            'fields[0].type must be exactly "vector"',
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": [_vector_field(path="  ")]},
            "fields[0].path must be a non-empty string",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": [_vector_field(path=0)]},
            "fields[0].path must be a non-empty string",
        ),
        (
            {
                "database_name": "db",
                "model_name": "mini",
                "fields": [_vector_field(similarity="euclidean")],
            },
            "euclidean and dotProduct are not supported",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": [_vector_field(similarity="")]},
            "similarity is required and must be cosine",
        ),
        (
            {"database_name": "db", "model_name": "mini", "fields": [_vector_field(similarity=1)]},
            "similarity is required and must be cosine",
        ),
    ],
)
def test_post_embedding_model_rejects_invalid_identity_and_fields(payload, error_snippet: str) -> None:
    mongo = MagicMock()
    collection = _collection(mongo)
    client = _client(mongo)

    response = client.post("/embedding-models", json=payload)

    assert response.status_code == 400
    assert error_snippet in response.get_json()["error"]
    collection.update_one.assert_not_called()


def test_post_embedding_model_keeps_uppercase_cosine_similarity() -> None:
    """The cosine check is case-insensitive, but the stored similarity is not rewritten."""
    mongo = MagicMock()
    collection = _collection(mongo)
    client = _client(mongo)
    payload = {
        "database_name": "privacy",
        "model_name": "mini",
        "fields": [_vector_field(similarity="COSINE")],
    }

    with patch("util.utc_now", return_value="stamp"):
        response = client.post("/embedding-models", json=payload)

    assert response.status_code == 200
    stored = response.get_json()["fields"][0]["similarity"]
    assert stored == "COSINE"
    assert collection.update_one.call_args.args[1]["$set"]["fields"][0]["similarity"] == "COSINE"


def test_post_embedding_model_empty_list_body_is_treated_as_missing() -> None:
    """[] is falsy, so get_json() or {} runs before the object check."""
    mongo = MagicMock()
    collection = _collection(mongo)
    client = _client(mongo)

    response = client.post("/embedding-models", json=[])

    assert response.status_code == 400
    assert "database_name is required" in response.get_json()["error"]
    collection.update_one.assert_not_called()


@pytest.mark.parametrize("body", [["not-empty"], "not-an-object"])
def test_post_embedding_model_rejects_non_object_body(body) -> None:
    mongo = MagicMock()
    collection = _collection(mongo)
    client = _client(mongo)

    response = client.post("/embedding-models", json=body)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must be a JSON object"
    collection.update_one.assert_not_called()


def test_get_embedding_models_empty_name_is_unfiltered_whitespace_is_exact() -> None:
    mongo = MagicMock()
    collection = _collection(mongo)
    collection.find.return_value = []
    client = _client(mongo)

    empty = client.get("/embedding-models?database_name=")
    padded = client.get("/embedding-models?database_name=%20%20")

    assert empty.status_code == 200
    assert padded.status_code == 200
    assert collection.find.call_args_list[0].args == ({}, {"_id": 0})
    assert collection.find.call_args_list[1].args == ({"database_name": "  "}, {"_id": 0})
