"""Leftover POST /vector-search query_vector numeric-coercion failures."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def test_vector_search_query_vector_non_numeric_rejected(client) -> None:
    """Non-numeric entries fail float() and are 400 before aggregate."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [1, "x"],
            "path": "embedding",
        },
    )

    assert response.status_code == 400
    assert "only numbers" in response.get_json()["error"]


def test_vector_search_query_vector_none_entry_rejected(client) -> None:
    """None cannot be coerced with float() and is 400 before aggregate."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [None],
            "path": "embedding",
        },
    )

    assert response.status_code == 400
    assert "only numbers" in response.get_json()["error"]
