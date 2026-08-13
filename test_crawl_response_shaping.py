"""Regression tests for /crawl response shaping after a crawler returns pages.

Base route tests only assert crawl_method and pages_crawled. The endpoint also
formats combined_text, extracts urls_crawled, and maps empty-content / total
failure into 400/500. Those branches are independent of BFS helper coverage.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.register_blueprint(core_bp)
    return flask_app


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


def test_crawl_browser_combined_text_and_urls(client, mock_clients) -> None:
    """Browser dict pages are joined with title|Source headers and url list."""
    pages = [
        {"url": "https://example.com/a", "title": "Alpha", "markdown": "AAA"},
        {"url": "https://example.com/b", "title": "", "markdown": "BBB"},
    ]
    with patch("core.playwright_crawl", return_value=pages):
        response = client.post("/crawl", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "playwright"
    assert payload["pages_crawled"] == 2
    assert payload["urls_crawled"] == ["https://example.com/a", "https://example.com/b"]
    assert "Alpha | Source: https://example.com/a" in payload["combined_text"]
    assert payload["combined_text"].index("AAA") < payload["combined_text"].index("BBB")
    assert "Source: https://example.com/b" in payload["combined_text"]
    assert payload["text_length"] == len(payload["combined_text"])


def test_crawl_empty_markdown_returns_no_content(client, mock_clients) -> None:
    """A crawler that returns pages without body text is a 400, not a success."""
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": "https://example.com", "title": "T", "markdown": ""}],
    ):
        response = client.post("/crawl", json={"url": "https://example.com"})

    assert response.status_code == 400
    assert "no content" in response.get_json()["error"]


def test_crawl_all_methods_fail_returns_500_with_aggregated_errors(client, mock_clients) -> None:
    """When browsers and Firecrawl all fail, the 500 includes each method's error."""
    mock_clients["firecrawl"].crawl.side_effect = RuntimeError("firecrawl down")
    with (
        patch("core.playwright_crawl", side_effect=RuntimeError("pw failed")),
        patch("core.puppeteer_crawl", side_effect=RuntimeError("pp failed")),
        patch("core.selenium_crawl", side_effect=RuntimeError("se failed")),
    ):
        response = client.post("/crawl", json={"url": "https://example.com"})

    assert response.status_code == 500
    error = response.get_json()["error"]
    assert "playwright" in error
    assert "puppeteer" in error
    assert "selenium" in error
    assert "firecrawl" in error
    assert "pw failed" in error
    assert "firecrawl down" in error


def test_crawl_firecrawl_metadata_urls(client, mock_clients) -> None:
    """Firecrawl Document-like objects expose URL via metadata, not a top-level key."""
    page = MagicMock()
    page.metadata = MagicMock()
    page.metadata.url = "https://example.com/doc"
    page.metadata.title = "Doc"
    page.markdown = "Body from firecrawl"
    page.html = ""
    page.raw_html = ""
    page.summary = ""
    mock_clients["firecrawl"].crawl.return_value = [page]

    with (
        patch("core.playwright_crawl", return_value=[]),
        patch("core.puppeteer_crawl", return_value=[]),
        patch("core.selenium_crawl", return_value=[]),
    ):
        response = client.post("/crawl", json={"url": "https://example.com/doc"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "firecrawl"
    assert payload["urls_crawled"] == ["https://example.com/doc"]
    assert "Body from firecrawl" in payload["combined_text"]
    assert "Doc | Source: https://example.com/doc" in payload["combined_text"]
