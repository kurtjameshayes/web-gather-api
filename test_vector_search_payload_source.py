"""Regression tests for leftover /vector-search payload-source and coercion contracts.

POST without JSON Content-Type currently reads query args (the body is ignored).
query_vector entries are coerced with float(), so numeric strings and bools pass.
A JSON-string filter that decodes to a list is rejected after parse, distinct from
a list-typed filter which is rejected before json.loads.
"""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from conftest import _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def mock_clients() -> dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients: dict[str, Any]):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_aggregate(mock_clients: dict[str, Any]) -> MagicMock:
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = [{"_id": "1", "text": "hit"}]
    return mock_coll


def test_vector_search_post_without_json_uses_query_string(
    client, mock_clients
) -> None:
    """POST /vector-search with no JSON Content-Type reads query args, not the body."""
    mock_coll = _wire_aggregate(mock_clients)
    query_vector = json.dumps([0.1, 0.2])
    response = client.post(
        "/vector-search"
        "?database=db&collection=coll&index=idx"
        f"&query_vector={query_vector}&path=embedding"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["database"] == "db"
    assert data["collection"] == "coll"
    assert data["index"] == "idx"
    assert mock_coll.aggregate.called
    pipeline = mock_coll.aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["queryVector"] == [0.1, 0.2]


def test_vector_search_post_non_json_body_is_ignored(client, mock_clients) -> None:
    """A non-JSON Content-Type body is ignored even when it contains valid params."""
    _wire_aggregate(mock_clients)
    body = json.dumps(
        {
            "database": "from-body",
            "collection": "from-body",
            "index": "from-body",
            "query_vector": [9.9, 9.9],
            "path": "embedding",
        }
    )
    response = client.post(
        "/vector-search"
        "?database=from-query&collection=from-query&index=from-query"
        "&query_vector=%5B0.5%2C0.25%5D&path=embedding",
        data=body,
        content_type="text/plain",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["database"] == "from-query"
    assert data["collection"] == "from-query"
    assert data["index"] == "from-query"


def test_vector_search_post_non_json_without_query_is_400(client, mock_clients) -> None:
    """POST with a JSON-looking body but non-JSON Content-Type still misses required params."""
    response = client.post(
        "/vector-search",
        data=json.dumps(
            {
                "database": "db",
                "collection": "coll",
                "index": "idx",
                "query_vector": [0.1, 0.2],
                "path": "embedding",
            }
        ),
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_vector_search_query_vector_coerces_numeric_strings_and_bools(
    client, mock_clients
) -> None:
    """float() coercion accepts numeric strings and bools (True -> 1.0, False -> 0.0)."""
    mock_coll = _wire_aggregate(mock_clients)
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [True, False, "1.5", 2],
            "path": "embedding",
        },
    )
    assert response.status_code == 200
    pipeline = mock_coll.aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["queryVector"] == [1.0, 0.0, 1.5, 2.0]


def test_vector_search_filter_json_string_array_is_400(client, mock_clients) -> None:
    """A JSON-string filter that decodes to a list is 400 after parse (must be an object)."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
            "filter": "[1, 2]",
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "filter must be a JSON object"
