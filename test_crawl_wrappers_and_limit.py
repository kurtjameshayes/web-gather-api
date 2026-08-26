"""Regression tests for playwright/puppeteer sync wrappers and vector-search limit fallback."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

import core
from core import core_bp, init_core
from conftest import _wire_mongo_core


def test_playwright_crawl_wrapper_delegates_to_async_impl() -> None:
    """playwright_crawl runs the async crawler on a dedicated event loop."""
    pages = [{"url": "https://example.com", "title": "T", "markdown": "Body"}]
    with patch(
        "core._playwright_crawl_async",
        new=AsyncMock(return_value=pages),
    ) as mock_async:
        result = core.playwright_crawl("https://example.com", 2, 5)
    assert result == pages
    mock_async.assert_awaited_once_with("https://example.com", 2, 5)


def test_puppeteer_crawl_wrapper_delegates_to_async_impl() -> None:
    """puppeteer_crawl runs the async crawler on a dedicated event loop."""
    pages = [{"url": "https://example.com/p", "title": "T", "markdown": "Body"}]
    with patch(
        "core._puppeteer_crawl_async",
        new=AsyncMock(return_value=pages),
    ) as mock_async:
        result = core.puppeteer_crawl("https://example.com/p", 1, 3)
    assert result == pages
    mock_async.assert_awaited_once_with("https://example.com/p", 1, 3)


@pytest.fixture
def app():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    _wire_mongo_core(mock_mongo)
    flask_app = Flask(__name__)
    flask_app.register_blueprint(core_bp)
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_vector_search_invalid_limit_falls_back_to_default(client) -> None:
    """Non-integer limit is treated as 10, not a 400."""
    response = client.post(
        "/vector-search",
        json={
            "database": "db",
            "collection": "coll",
            "index": "idx",
            "query_vector": [0.1, 0.2],
            "path": "embedding",
            "limit": "not-a-number",
        },
    )
    assert response.status_code == 200
    from core import mongo_client

    pipeline = mongo_client["db"]["coll"].aggregate.call_args[0][0]
    assert pipeline[0]["$vectorSearch"]["limit"] == 10
    assert pipeline[0]["$vectorSearch"]["numCandidates"] == 100
