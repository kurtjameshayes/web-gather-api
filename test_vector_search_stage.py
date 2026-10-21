"""POST /vector-search stage assembly beyond the happy-path endpoint tests.

Pins candidate count, vector redaction, model selection, and rejection
order. These decide what Atlas searches and what embeddings leave the API.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId
from flask import Flask

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
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _coll(mock_clients, rows=None, error=None):
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    if error is not None:
        coll.aggregate.side_effect = error
    else:
        coll.aggregate.return_value = rows or []
    return coll


def _stage(coll):
    pipeline = coll.aggregate.call_args[0][0]
    return pipeline[0]["$vectorSearch"]


@pytest.mark.parametrize(
    ("limit", "expected_limit", "expected_candidates"),
    [
        (1, 1, 100),
        (20, 20, 200),
        (100, 100, 1000),
        (101, 10, 100),
    ],
)
def test_num_candidates_uses_reset_limit_and_floor(
    client, mock_clients, limit, expected_limit, expected_candidates
):
    coll = _coll(mock_clients)
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [0.5, 1],
            "path": "embedding",
            "limit": limit,
        },
    )
    assert response.status_code == 200
    stage = _stage(coll)
    assert stage["limit"] == expected_limit
    assert stage["numCandidates"] == expected_candidates
    assert stage["queryVector"] == [0.5, 1.0]
    assert "filter" not in stage
    assert "query" not in response.get_json()


def test_empty_query_is_echoed_when_a_vector_is_present(client, mock_clients):
    _coll(mock_clients)
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
            "query": "",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["query"] == ""


def test_filter_shapes(client, mock_clients):
    coll = _coll(mock_clients)
    array_filter = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
            "filter": [1],
        },
    )
    assert array_filter.status_code == 400
    assert array_filter.get_json()["error"] == "filter must be a JSON object or JSON string"

    parsed_array = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
            "filter": "[]",
        },
    )
    assert parsed_array.status_code == 400
    assert parsed_array.get_json()["error"] == "filter must be a JSON object"
    coll.aggregate.assert_not_called()

    empty = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
            "filter": {},
        },
    )
    assert empty.status_code == 200
    assert "filter" not in _stage(coll)

    encoded = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
            "filter": '{"jurisdiction": "CA"}',
        },
    )
    assert encoded.status_code == 200
    assert _stage(coll)["filter"] == {"jurisdiction": "CA"}


def test_response_strips_only_the_configured_path_and_stringifies_id(client, mock_clients):
    oid = ObjectId()
    coll = _coll(
        mock_clients,
        rows=[
            {
                "_id": oid,
                "embedding": [9, 9],
                "other_vector": [3],
                "text": "kept",
            }
        ],
    )
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
        },
    )
    assert response.status_code == 200
    row = response.get_json()["results"][0]
    assert row["_id"] == str(oid)
    assert "embedding" not in row
    assert row["other_vector"] == [3]
    assert row["text"] == "kept"

    padded = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": " embedding ",
        },
    )
    assert padded.status_code == 200
    leaked = padded.get_json()["results"][0]
    assert leaked["embedding"] == [9, 9]
    assert _stage(coll)["path"] == " embedding "


def test_blank_first_field_path_does_not_use_a_later_field(client, mock_clients):
    coll = _coll(mock_clients)
    with patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": ""}, {"path": "embedding"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": "db",
                "collection": "chunks",
                "index": "idx",
                "query_vector": [1],
            },
        )
    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Vector field path unknown.")
    coll.aggregate.assert_not_called()


def test_aggregate_error_includes_the_exception_text(client, mock_clients):
    _coll(mock_clients, error=RuntimeError("disk down"))
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "idx",
            "query_vector": [1],
            "path": "embedding",
        },
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Vector search failed: disk down"


def test_query_vector_skips_model_lookup(client, mock_clients):
    _coll(mock_clients)
    with patch("core.get_model") as get_model, patch(
        "core.get_application_embedding_model"
    ) as app_model, patch("core.get_embedding_model_name") as per_db:
        response = client.post(
            "/vector-search",
            json={
                "database": PRIVACY_COMPLIANCE_DB,
                "collection": "chunks",
                "index": "idx",
                "query_vector": [0.2],
                "path": "embedding",
            },
        )
    assert response.status_code == 200
    get_model.assert_not_called()
    app_model.assert_not_called()
    per_db.assert_not_called()


def test_privacy_query_uses_application_model_including_whitespace(client, mock_clients):
    _coll(mock_clients, rows=[{"_id": "1", "embedding": [1], "text": "t"}])
    with patch("core.get_application_embedding_model", return_value="  mini  ") as app_model, patch(
        "core.get_embedding_model_name"
    ) as per_db, patch("core.get_model", return_value=DummyModel()) as get_model, patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "embedding"}]},
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": PRIVACY_COMPLIANCE_DB,
                "collection": "chunks",
                "index": "idx",
                "query": "hello",
            },
        )
    assert response.status_code == 200
    assert "embedding" not in response.get_json()["results"][0]
    get_model.assert_called_once_with("  mini  ")
    app_model.assert_called_once()
    per_db.assert_not_called()


def test_blank_application_model_falls_back_and_other_databases_skip_it(client, mock_clients):
    _coll(mock_clients)
    with patch("core.get_application_embedding_model", return_value="") as app_model, patch(
        "core.get_embedding_model_name", return_value="db-model"
    ) as per_db, patch("core.get_model", return_value=DummyModel()) as get_model, patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "vec"}]},
    ):
        fallback = client.post(
            "/vector-search",
            json={
                "database": PRIVACY_COMPLIANCE_DB,
                "collection": "chunks",
                "index": "idx",
                "query": "hello",
            },
        )
    assert fallback.status_code == 200
    get_model.assert_called_once_with("db-model")
    per_db.assert_called_once_with(PRIVACY_COMPLIANCE_DB)
    app_model.assert_called_once()

    with patch("core.get_application_embedding_model") as app_model, patch(
        "core.get_embedding_model_name", return_value="other-model"
    ) as per_db, patch("core.get_model", return_value=DummyModel()) as get_model, patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "vec"}]},
    ):
        other = client.post(
            "/vector-search",
            json={
                "database": "other-db",
                "collection": "chunks",
                "index": "idx",
                "query": "hello",
            },
        )
    assert other.status_code == 200
    app_model.assert_not_called()
    per_db.assert_called_once_with("other-db")
    get_model.assert_called_once_with("other-model")
