"""Regression tests for /crawl and /ingest depth/breadth validation and clamps."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import WEB_GATHER_DB, core_bp, init_core


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
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_mongo(mock_mongo: MagicMock) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()


def test_crawl_clamps_depth_and_breadth(client, mock_clients) -> None:
    """Crawl depth is capped at 10 and breadth at 100 before invoking crawlers."""
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": "https://example.com", "title": "T", "markdown": "Body"}],
    ) as playwright_crawl:
        response = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 99, "breadth": 500},
        )

    assert response.status_code == 200
    playwright_crawl.assert_called_once_with("https://example.com", 10, 100)


def test_crawl_raises_floor_for_zero_depth_and_breadth(client, mock_clients) -> None:
    """Zero depth/breadth are raised to 1 (not rejected) to keep crawls usable."""
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": "https://example.com", "title": "T", "markdown": "Body"}],
    ) as playwright_crawl:
        response = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 0, "breadth": 0},
        )

    assert response.status_code == 200
    playwright_crawl.assert_called_once_with("https://example.com", 1, 1)


def test_crawl_rejects_non_integer_depth_and_breadth(client, mock_clients) -> None:
    """Non-integer depth/breadth must 400 before any crawler runs."""
    with patch("core.playwright_crawl") as playwright_crawl:
        bad_depth = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": "deep", "breadth": 5},
        )
        bad_breadth = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 2, "breadth": "wide"},
        )

    assert bad_depth.status_code == 400
    assert "depth" in bad_depth.get_json()["error"].lower()
    assert bad_breadth.status_code == 400
    assert "breadth" in bad_breadth.get_json()["error"].lower()
    playwright_crawl.assert_not_called()


def test_ingest_clamps_breadth_and_rejects_invalid_bounds(client, mock_clients) -> None:
    """Ingest breadth is capped at 100; invalid depth/breadth types return 400."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com", "title": "T", "markdown": "Hello"}
    ]

    with patch("core.detect_content_type", return_value=""):
        ok = client.post(
            "/ingest",
            json={
                "url": "https://example.com",
                "database": "test_db",
                "collection": "docs",
                "depth": 0,
                "breadth": 250,
            },
        )
        bad_depth = client.post(
            "/ingest",
            json={
                "url": "https://example.com",
                "database": "test_db",
                "collection": "docs",
                "depth": "nope",
            },
        )
        bad_breadth = client.post(
            "/ingest",
            json={
                "url": "https://example.com",
                "database": "test_db",
                "collection": "docs",
                "breadth": "nope",
            },
        )

    assert ok.status_code == 200
    mock_clients["firecrawl"].crawl.assert_called_with(
        url="https://example.com",
        limit=100,
    )
    assert bad_depth.status_code == 400
    assert "depth" in bad_depth.get_json()["error"].lower()
    assert bad_breadth.status_code == 400
    assert "breadth" in bad_breadth.get_json()["error"].lower()
