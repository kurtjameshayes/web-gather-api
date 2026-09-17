"""POST /vector-search limit reset and empty-vector-with-query contracts.

Nearby coverage resets limit=999 to 10. Out-of-range values (0, negative) also
reset to 10 rather than clamping like /documents or /runs. An empty
query_vector with a query reaches the dedicated empty-list error instead of
the missing-params branch used when both are omitted.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core  # noqa: E402


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def app(mock_clients) -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def _wire_collection(mock_mongo: MagicMock) -> MagicMock:
    coll = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = db
    return coll


def _search_body(**overrides: Any) -> dict:
    body = {
        "database": "db",
        "collection": "chunks",
        "index": "vector_idx",
        "query_vector": [0.1, 0.2],
        "path": "embedding",
    }
    body.update(overrides)
    return body


def test_vector_search_zero_and_negative_limit_reset_to_default(
    client, mock_clients
) -> None:
    """limit<1 resets to 10; it is not clamped to 1 like GET /documents."""
    coll = _wire_collection(mock_clients["mongo"])
    coll.aggregate.return_value = []

    zero = client.post("/vector-search", json=_search_body(limit=0))
    assert zero.status_code == 200
    stage = coll.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert stage["limit"] == 10
    assert stage["numCandidates"] == 100

    negative = client.post("/vector-search", json=_search_body(limit=-4))
    assert negative.status_code == 200
    stage = coll.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert stage["limit"] == 10


def test_vector_search_invalid_limit_string_resets_to_default(
    client, mock_clients
) -> None:
    """Non-integer limit falls back to 10 before the Atlas aggregate."""
    coll = _wire_collection(mock_clients["mongo"])
    coll.aggregate.return_value = []

    response = client.post("/vector-search", json=_search_body(limit="abc"))
    assert response.status_code == 200
    stage = coll.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert stage["limit"] == 10
    assert stage["numCandidates"] == 100


def test_vector_search_empty_query_vector_with_query_is_empty_error(
    client, mock_clients
) -> None:
    """query plus [] is 400 'must not be empty', not missing query_vector."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "vector_idx",
            "query": "retention",
            "query_vector": [],
            "path": "embedding",
        },
    )
    assert response.status_code == 400
    error = response.get_json()["error"].lower()
    assert "empty" in error
    assert "missing required parameters" not in error
