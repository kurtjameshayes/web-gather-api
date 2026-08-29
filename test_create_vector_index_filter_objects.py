"""Regression tests for POST /create-vector-index object filters and defaults."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from util import VECTOR_INDEX_SIMILARITY, init_util, util_bp


@pytest.fixture
def target_db() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(target_db: MagicMock):
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def _record_fields():
    return [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "dotProduct",
        },
        {"type": "filter", "path": "already_there"},
    ]


def test_create_vector_index_object_filters_and_default_name(client, target_db) -> None:
    """Object filter_fields are accepted; blank paths skipped; omitted name is vector_index."""
    target_db.command.side_effect = [
        {"ok": 1},  # drop
        {"ok": 1},  # create
    ]
    record = {"database_name": "test_db", "model_name": "m", "fields": _record_fields()}

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "filter_fields": [
                    {"path": "jurisdiction"},
                    {"path": "   "},
                    {"path": " document_id "},
                    {"path": 123},
                    "  ",
                ],
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["index_name"] == "vector_index"
    assert data["filter_fields_added"] == 3

    create_cmd = target_db.command.call_args_list[1][0][0]
    assert create_cmd["createSearchIndexes"] == "coll"
    fields = create_cmd["indexes"][0]["definition"]["fields"]
    vector = next(f for f in fields if isinstance(f, dict) and f.get("type") == "vector")
    assert vector["similarity"] == VECTOR_INDEX_SIMILARITY == "cosine"
    # Non-vector record fields are copied through (not dropped).
    assert {"type": "filter", "path": "already_there"} in fields
    assert {"type": "filter", "path": "jurisdiction"} in fields
    assert {"type": "filter", "path": "document_id"} in fields
    assert {"type": "filter", "path": "123"} in fields
    assert not any(isinstance(f, dict) and f.get("path") == "" for f in fields)


def test_create_vector_index_omitted_filters_do_not_drop_non_vector_fields(client, target_db) -> None:
    """When request omits filter_fields, non-vector embedding_model fields still remain."""
    target_db.command.side_effect = [Exception("missing"), {"ok": 1}]
    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": _record_fields(),
        "filter_fields": [{"path": "company_name"}],
    }

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["index_name"] == "vector_index"
    assert data["filter_fields_added"] == 1
    create_cmd = target_db.command.call_args_list[1][0][0]
    fields = create_cmd["indexes"][0]["definition"]["fields"]
    assert {"type": "filter", "path": "already_there"} in fields
    assert {"type": "filter", "path": "company_name"} in fields
