"""SSRF decisions for ranges earlier samples do not name.

Pins NAT64/DNS64 discovery, the rest of 0.0.0.0/8, IPv6 benchmarking,
the ORCHIDv1/v2 boundary, and IPv4-mapped cloud metadata.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core, _validate_url_block_ssrf


PRIVATE = "URL must not target private or link-local addresses"


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


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.0.170/nat64-discovery",
        "http://192.0.0.171/dns64-discovery",
        "http://user:pass@0.0.0.1/this-network",
        "http://[2001:2::1]/benchmark",
        "http://[2001:10::1]/orchid-v1",
        "http://[::ffff:169.254.169.254]/latest/meta-data",
    ],
)
def test_special_use_ranges_are_blocked(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (False, PRIVATE)


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.0.10/turn-anycast",
        "http://[2001:20::1]/orchid-v2",
    ],
)
def test_neighbor_ranges_stay_reachable(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (True, None)


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.0.170/nat64-discovery",
        "http://[2001:2::1]/benchmark",
    ],
)
def test_crawl_blocks_special_use_before_fetch(client, mock_clients, url: str) -> None:
    with patch("core.playwright_crawl") as playwright, patch(
        "core.puppeteer_crawl"
    ) as puppeteer, patch("core.selenium_crawl") as selenium:
        response = client.post("/crawl", json={"url": url})

    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    playwright.assert_not_called()
    puppeteer.assert_not_called()
    selenium.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.1/this-network",
        "http://[::ffff:169.254.169.254]/latest/meta-data",
    ],
)
def test_ingest_blocks_special_use_before_fetch(client, mock_clients, url: str) -> None:
    with patch("core.requests.head") as head, patch("core.requests.get") as get:
        response = client.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    head.assert_not_called()
    get.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_crawl_allows_turn_anycast_and_uses_playwright_defaults(client, mock_clients) -> None:
    url = "http://192.0.0.10/turn-anycast"
    pages = [{"url": url, "title": "Turn", "markdown": "public relay"}]
    with patch("core.playwright_crawl", return_value=pages) as playwright:
        response = client.post("/crawl", json={"url": url})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "playwright"
    assert payload["depth"] == 2
    assert payload["breadth"] == 10
    assert payload["combined_text"] == f"Turn | Source: {url}\npublic relay"
    playwright.assert_called_once_with(url, 2, 10)
    mock_clients["firecrawl"].crawl.assert_not_called()
