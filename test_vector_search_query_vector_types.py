"""Regression tests for /vector-search query_vector and filter type checks."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client() -> Any:
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_vector_search_query_vector_must_contain_only_numbers(client) -> None:
    """Non-numeric query_vector entries are rejected before embedding or aggregate."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [0.1, "nope"],
            "path": "embedding",
        },
    )

    assert response.status_code == 400
    assert "numbers" in response.get_json()["error"].lower()


def test_vector_search_filter_must_be_object_or_json_string(client) -> None:
    """A list filter is rejected; only objects or JSON object strings are allowed."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
            "filter": ["jurisdiction", "CA"],
        },
    )

    assert response.status_code == 400
    assert "filter" in response.get_json()["error"].lower()
