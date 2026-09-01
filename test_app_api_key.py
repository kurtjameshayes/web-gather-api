"""Regression tests for APP_API_KEY gating on core/db/util routes."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

import security
from db import db_bp, init_db
from security import init_app_api_key, require_api_key_async

# core.py imports pyppeteer at module load; stub before importing vector-search.
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())


@pytest.fixture
def restore_app_api_key():
    """Isolate the module-level APP_API_KEY flag across tests."""
    original = security._app_api_key
    yield
    security._app_api_key = original


@pytest.fixture
def db_client():
    mock_mongo = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count_documents.return_value = 3
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_collection
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_init_app_api_key_unset_is_none(monkeypatch, restore_app_api_key) -> None:
    monkeypatch.delenv("APP_API_KEY", raising=False)
    init_app_api_key()
    assert security._app_api_key is None


def test_init_app_api_key_whitespace_only_is_none(monkeypatch, restore_app_api_key) -> None:
    """Whitespace-only APP_API_KEY currently disables auth (strip then falsy)."""
    monkeypatch.setenv("APP_API_KEY", "   ")
    init_app_api_key()
    assert security._app_api_key is None


def test_init_app_api_key_strips_surrounding_whitespace(monkeypatch, restore_app_api_key) -> None:
    monkeypatch.setenv("APP_API_KEY", "  secret-key  ")
    init_app_api_key()
    assert security._app_api_key == "secret-key"


def test_require_api_key_passthrough_when_unset(db_client, restore_app_api_key) -> None:
    security._app_api_key = None
    response = db_client.get(
        "/count-documents?database_name=test_db&collection_name=docs"
    )
    assert response.status_code == 200
    assert response.get_json()["count"] == 3


def test_require_api_key_missing_header_is_401(db_client, restore_app_api_key) -> None:
    security._app_api_key = "secret-key"
    response = db_client.get(
        "/count-documents?database_name=test_db&collection_name=docs"
    )
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."


def test_require_api_key_whitespace_header_is_401(db_client, restore_app_api_key) -> None:
    security._app_api_key = "secret-key"
    response = db_client.get(
        "/count-documents?database_name=test_db&collection_name=docs",
        headers={"x-api-key": "   "},
    )
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."


def test_require_api_key_invalid_header_is_403(db_client, restore_app_api_key) -> None:
    security._app_api_key = "secret-key"
    response = db_client.get(
        "/count-documents?database_name=test_db&collection_name=docs",
        headers={"x-api-key": "wrong"},
    )
    assert response.status_code == 403
    assert response.get_json()["error"] == "Invalid API key."


def test_require_api_key_accepts_stripped_matching_header(db_client, restore_app_api_key) -> None:
    security._app_api_key = "secret-key"
    response = db_client.get(
        "/count-documents?database_name=test_db&collection_name=docs",
        headers={"x-api-key": "  secret-key  "},
    )
    assert response.status_code == 200
    assert response.get_json()["count"] == 3


def test_require_api_key_async_missing_and_invalid(restore_app_api_key) -> None:
    """Async decorator shares the same 401/403/pass contracts as the sync decorator."""
    security._app_api_key = "secret-key"
    app = Flask(__name__)

    @require_api_key_async
    async def protected():
        return {"ok": True}

    with app.test_request_context("/"):
        missing = asyncio.run(protected())
    assert missing[1] == 401
    assert missing[0].get_json()["error"] == "Missing API key."

    with app.test_request_context("/", headers={"x-api-key": "wrong"}):
        invalid = asyncio.run(protected())
    assert invalid[1] == 403
    assert invalid[0].get_json()["error"] == "Invalid API key."

    with app.test_request_context("/", headers={"x-api-key": "secret-key"}):
        ok = asyncio.run(protected())
    assert ok == {"ok": True}


def test_vector_search_is_not_gated_by_app_api_key(restore_app_api_key) -> None:
    """POST /vector-search currently has no @require_api_key decorator.

    With APP_API_KEY set, a missing key still reaches parameter validation (400)
    instead of 401. Pin this so adding or removing the decorator is intentional.
    """
    from core import core_bp, init_core

    security._app_api_key = "secret-key"
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    client = app.test_client()

    response = client.post("/vector-search", json={})
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]
