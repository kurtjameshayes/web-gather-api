"""Regression tests for POST /vector-search query echo vs query_vector-only."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from conftest import _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_vector_search_omits_query_when_only_query_vector(client, mock_clients) -> None:
    """``query`` is included only when the request provided it (even if empty)."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    mock_coll = dbs["source_db"].__getitem__.return_value
    mock_coll.aggregate.return_value = [{"_id": "1", "text": "match"}]

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
    assert "query" not in data
    assert data["results"][0]["text"] == "match"
