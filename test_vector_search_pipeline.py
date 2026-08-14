"""Regression tests for /vector-search limit clamping, filters, and failures.

Existing tests cover missing params, query_vector parsing, and stripping the
embedding field. Limit sanitization, filter injection into $vectorSearch, _id
stringification, and aggregate 500s were still untested on v0_1.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


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


def test_vector_search_clamps_invalid_limit_to_default(client, mock_clients) -> None:
    coll = _wire_collection(mock_clients["mongo"])
    coll.aggregate.return_value = []

    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
            "limit": 999,
        },
    )
    assert response.status_code == 200
    stage = coll.aggregate.call_args[0][0][0]["$vectorSearch"]
    assert stage["limit"] == 10
    assert stage["numCandidates"] == 100


def test_vector_search_applies_filter_and_stringifies_id(client, mock_clients) -> None:
    coll = _wire_collection(mock_clients["mongo"])
    oid = ObjectId()
    coll.aggregate.return_value = [
        {"_id": oid, "text": "hit", "embedding": [0.1, 0.2], "score": 0.88}
    ]

    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
            "limit": 3,
            "filter": {"jurisdiction": "CA"},
        },
    )
    assert response.status_code == 200
    pipeline = coll.aggregate.call_args[0][0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["filter"] == {"jurisdiction": "CA"}
    assert stage["limit"] == 3
    assert stage["numCandidates"] == 100
    result = response.get_json()["results"][0]
    assert result["_id"] == str(oid)
    assert "embedding" not in result
    assert result["text"] == "hit"


def test_vector_search_aggregate_failure_returns_500(client, mock_clients) -> None:
    coll = _wire_collection(mock_clients["mongo"])
    coll.aggregate.side_effect = RuntimeError("index missing")

    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
        },
    )
    assert response.status_code == 500
    assert "vector search failed" in response.get_json()["error"].lower()
