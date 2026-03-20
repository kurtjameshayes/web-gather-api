"""Tests for core /vector-search endpoint (GET and POST)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

from core import PRIVACY_COMPLIANCE_DB, core_bp, init_core
from conftest import _wire_mongo_core, DummyModel


@pytest.fixture
def mock_clients():
    """Mock MongoDB, Firecrawl, Anthropic for core."""
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    """Flask app with core blueprint."""
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    """Test client."""
    return app.test_client()


def test_vector_search_post_success_with_query_vector(client, mock_clients) -> None:
    """POST /vector-search with query_vector and path returns results."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = [
        {"_id": "1", "text": "match", "score": 0.9},
    ]

    response = client.post(
        "/vector-search",
        json={
            "database": "test_db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": [0.1, 0.2, 0.3],
            "path": "embedding",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["database"] == "test_db"
    assert data["collection"] == "chunks"
    assert data["index"] == "vector_idx"
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "match"


def test_vector_search_get_success_with_query_vector(client, mock_clients) -> None:
    """GET /vector-search with query params works (query_vector as JSON string)."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = []

    response = client.get(
        "/vector-search?database=db&collection=coll&index=idx"
        "&query_vector=%5B0.1%2C0.2%5D&path=embedding"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["database"] == "db"
    assert data["collection"] == "coll"


def test_vector_search_missing_params(client, mock_clients) -> None:
    """POST /vector-search with missing params returns 400."""
    response = client.post("/vector-search", json={"database": "db"})
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_vector_search_invalid_filter_json(client, mock_clients) -> None:
    """POST /vector-search with invalid filter JSON returns 400."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [0.1],
            "path": "embedding",
            "filter": "{invalid}",
        },
    )
    assert response.status_code == 400
    assert "filter" in response.get_json()["error"].lower()


def test_vector_search_no_embedding_model(client, mock_clients) -> None:
    """POST /vector-search with query (no query_vector) when no model returns 400."""
    _wire_mongo_core(mock_clients["mongo"])
    with patch("core.get_embedding_model_name", return_value=None):
        response = client.post(
            "/vector-search",
            json={
                "database": "other_db",
                "collection": "coll",
                "index": "idx",
                "query": "search text",
            },
        )
    assert response.status_code == 400
    assert "embedding model" in response.get_json()["error"].lower()


def test_vector_search_no_vector_path(client, mock_clients) -> None:
    """POST /vector-search when path unknown returns 400."""
    _wire_mongo_core(mock_clients["mongo"])
    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_embedding_model_record", return_value=None
    ):
        response = client.post(
            "/vector-search",
            json={
                "database": "db",
                "collection": "coll",
                "index": "idx",
                "query": "text",
            },
        )
    assert response.status_code == 400
    assert "path" in response.get_json()["error"].lower() or "vector" in response.get_json()["error"].lower()


def test_vector_search_success_with_query(client, mock_clients) -> None:
    """POST /vector-search with query uses model and returns results."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = [
        {"_id": "1", "text": "result", "embedding": [0.1, 0.2]},
    ]

    with patch("core.get_embedding_model_name", return_value="model"), patch(
        "core.get_embedding_model_record",
        return_value={"fields": [{"path": "embedding", "numDimensions": 384}]},
    ), patch("core.get_model", return_value=DummyModel()):
        response = client.post(
            "/vector-search",
            json={
                "database": "db",
                "collection": "coll",
                "index": "idx",
                "query": "search text",
                "path": "embedding",
            },
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["query"] == "search text"
    assert len(data["results"]) == 1
    assert "embedding" not in data["results"][0]


def test_vector_search_query_vector_invalid_json(client, mock_clients) -> None:
    """POST /vector-search with query_vector as invalid JSON string returns 400."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": "not-json",
            "path": "embedding",
        },
    )
    assert response.status_code == 400
    assert "query_vector" in response.get_json()["error"].lower()


def test_vector_search_query_vector_not_list(client, mock_clients) -> None:
    """POST /vector-search with query_vector as object returns 400."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": {"a": 1},
            "path": "embedding",
        },
    )
    assert response.status_code == 400
    assert "list" in response.get_json()["error"].lower()


def test_vector_search_query_vector_empty(client, mock_clients) -> None:
    """POST /vector-search with empty query_vector returns 400."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [],
            "path": "embedding",
        },
    )
    assert response.status_code == 400
    # Empty list is falsy so may be treated as missing; either way we get 400
    err = response.get_json()["error"].lower()
    assert "query" in err or "empty" in err
