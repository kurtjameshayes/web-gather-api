"""Regression tests for Firecrawl call contracts on /ingest vs /crawl.

/ingest crawls with url + limit=breadth only. /crawl's Firecrawl fallback also
passes max_discovery_depth=depth. Swapping those kwargs would change crawl
breadth vs depth behavior.
"""
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


def _wire_mongo(mock_mongo: MagicMock) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()


def test_ingest_firecrawl_uses_limit_not_discovery_depth(client, mock_clients) -> None:
    """POST /ingest passes url and limit=breadth; it does not pass max_discovery_depth."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/page", "title": "T", "markdown": "Hello"}
    ]
    with patch("core.detect_content_type", return_value=""):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/page",
                "database": "test_db",
                "collection": "docs",
                "depth": 4,
                "breadth": 7,
            },
        )
    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    kwargs = mock_clients["firecrawl"].crawl.call_args.kwargs
    assert kwargs["url"] == "https://example.com/page"
    assert kwargs["limit"] == 7
    assert "max_discovery_depth" not in kwargs


def test_ingest_head_failure_still_crawls_as_web(client, mock_clients) -> None:
    """HEAD failures in detect_content_type must not treat a non-.pdf URL as PDF."""
    _wire_mongo(mock_clients["mongo"])
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com/article", "title": "T", "markdown": "Body"}
    ]
    with patch("core.requests.head", side_effect=ConnectionError("timed out")):
        response = client.post(
            "/ingest",
            json={
                "url": "https://example.com/article",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["document_type"] == "web"
    mock_clients["firecrawl"].crawl.assert_called_once()


def test_crawl_firecrawl_fallback_passes_max_discovery_depth(client, mock_clients) -> None:
    """POST /crawl Firecrawl fallback includes max_discovery_depth=depth."""
    mock_clients["firecrawl"].crawl.return_value = [
        {"url": "https://example.com", "title": "T", "markdown": "Body"}
    ]
    with patch("core.playwright_crawl", return_value=[]), patch(
        "core.puppeteer_crawl", return_value=[]
    ), patch("core.selenium_crawl", return_value=[]):
        response = client.post(
            "/crawl",
            json={"url": "https://example.com", "depth": 3, "breadth": 8},
        )
    assert response.status_code == 200
    assert response.get_json()["crawl_method"] == "firecrawl"
    kwargs = mock_clients["firecrawl"].crawl.call_args.kwargs
    assert kwargs["url"] == "https://example.com"
    assert kwargs["limit"] == 8
    assert kwargs["max_discovery_depth"] == 3
