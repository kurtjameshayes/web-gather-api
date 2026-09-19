"""POST /vector-search currently has no @require_api_key decorator.

With APP_API_KEY configured, decorated core routes (e.g. GET /search) return
401/403. /vector-search still validates params first, so a missing or wrong
key is 400 missing-params, not 401/403. Pin the current contract; do not add
the decorator in a coverage-only change.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

import security
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


@pytest.fixture
def configured_api_key():
    previous = security._app_api_key
    security._app_api_key = "secret-key"
    try:
        yield "secret-key"
    finally:
        security._app_api_key = previous


def test_vector_search_missing_key_is_400_not_401(client, configured_api_key) -> None:
    """Missing x-api-key on /vector-search is 400 missing-params, not 401."""
    response = client.post("/vector-search", json={})
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_vector_search_invalid_key_is_400_not_403(client, configured_api_key) -> None:
    """Wrong x-api-key on /vector-search is still 400, not 403 Invalid API key."""
    response = client.post(
        "/vector-search",
        json={},
        headers={"x-api-key": "wrong-key"},
    )
    assert response.status_code == 400
    err = response.get_json()["error"]
    assert "Missing required parameters" in err
    assert "Invalid API key" not in err


def test_search_missing_key_is_401_when_configured(client, configured_api_key) -> None:
    """GET /search is decorated, so a missing key is 401 when APP_API_KEY is set."""
    response = client.get("/search?document_id=doc-1&query=test")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."


def test_search_invalid_key_is_403_when_configured(client, configured_api_key) -> None:
    """GET /search with a mismatched key is 403."""
    response = client.get(
        "/search?document_id=doc-1&query=test",
        headers={"x-api-key": "wrong-key"},
    )
    assert response.status_code == 403
    assert response.get_json()["error"] == "Invalid API key."
