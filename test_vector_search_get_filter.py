"""Regression tests for GET /vector-search filter query-string parsing.

GET params are always strings, so filter must be JSON-decoded into an object
before it is attached to the $vectorSearch stage. POST JSON-object filters
are covered elsewhere.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import core_bp, init_core
from conftest import _wire_mongo_core


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


@pytest.fixture
def app(mock_clients) -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def test_vector_search_get_filter_json_string_applied_to_pipeline(
    client, mock_clients
) -> None:
    """GET filter JSON object string must land on $vectorSearch.filter."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = [
        {"_id": "1", "text": "ca match", "score": 0.8},
    ]

    query = urlencode(
        {
            "database": "test_db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": "[0.1, 0.2]",
            "path": "embedding",
            "filter": '{"jurisdiction": "CA"}',
        }
    )
    response = client.get(f"/vector-search?{query}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["results"][0]["text"] == "ca match"

    pipeline = mock_coll.aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["filter"] == {"jurisdiction": "CA"}
    assert pipeline[0]["$vectorSearch"]["path"] == "embedding"
    assert pipeline[0]["$vectorSearch"]["index"] == "vector_idx"


def test_vector_search_get_filter_array_json_rejected(client, mock_clients) -> None:
    """GET filter that JSON-decodes to a non-object must 400 before aggregating."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value

    query = urlencode(
        {
            "database": "test_db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": "[0.1, 0.2]",
            "path": "embedding",
            "filter": '["jurisdiction"]',
        }
    )
    response = client.get(f"/vector-search?{query}")
    assert response.status_code == 400
    assert "json object" in response.get_json()["error"].lower()
    mock_coll.aggregate.assert_not_called()


def test_vector_search_get_invalid_filter_json_rejected(client, mock_clients) -> None:
    """GET filter with malformed JSON must 400."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value

    query = urlencode(
        {
            "database": "test_db",
            "collection": "chunks",
            "index": "vector_idx",
            "query_vector": "[0.1, 0.2]",
            "path": "embedding",
            "filter": "{not-json}",
        }
    )
    response = client.get(f"/vector-search?{query}")
    assert response.status_code == 400
    assert "filter" in response.get_json()["error"].lower()
    mock_coll.aggregate.assert_not_called()
