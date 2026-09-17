"""Crawl/ingest JSON-null bounds and all-empty crawl fallback.

#205 clamps numeric 0 and rejects non-integer strings. JSON null is present so
the default is skipped, then int(None) is 400. All crawlers returning no pages
(without raising) is 400 'blocking crawlers', distinct from empty-markdown 400
and all-exception 500.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core  # noqa: E402


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


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


def test_crawl_null_depth_and_breadth_are_400_not_defaults(client, mock_clients) -> None:
    """Explicit JSON null skips the default and fails integer conversion."""
    with patch("core.playwright_crawl") as crawl_fn:
        bad_depth = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": None, "breadth": 5},
        )
        bad_breadth = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 2, "breadth": None},
        )

    assert bad_depth.status_code == 400
    assert "depth" in bad_depth.get_json()["error"].lower()
    assert bad_breadth.status_code == 400
    assert "breadth" in bad_breadth.get_json()["error"].lower()
    crawl_fn.assert_not_called()


def test_ingest_null_depth_is_400_before_crawl(client, mock_clients) -> None:
    """ingest depth=null is 400, unlike omitted default 1 or depth=0 which floors to 1."""
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com",
            "database": "test_db",
            "collection": "docs",
            "depth": None,
        },
    )
    assert response.status_code == 400
    assert "depth" in response.get_json()["error"].lower()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_crawl_all_empty_pages_is_400_blocking_not_500(client, mock_clients) -> None:
    """Browsers and Firecrawl returning no pages is 400, not an aggregated 500."""
    mock_clients["firecrawl"].crawl.return_value = []
    with (
        patch("core.playwright_crawl", return_value=[]),
        patch("core.puppeteer_crawl", return_value=[]),
        patch("core.selenium_crawl", return_value=[]),
    ):
        response = client.post("/crawl", json={"url": "https://example.com"})

    assert response.status_code == 400
    error = response.get_json()["error"].lower()
    assert "no pages" in error
    assert "blocking" in error
    mock_clients["firecrawl"].crawl.assert_called_once()
