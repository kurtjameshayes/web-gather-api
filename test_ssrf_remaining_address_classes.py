"""SSRF decisions for address classes not covered by earlier samples.

The guard classifies only hostnames that ipaddress.ip_address accepts, then
blocks private, loopback, and link-local addresses. These cases pin the
remaining split on Python 3.12: documentation and benchmark ranges are
private, while IPv6 multicast, deprecated site-local, 6to4 anycast, NAT64,
and non-literal loopback spellings are not classified as blocked IPs.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core, _validate_url_block_ssrf


PRIVATE_MESSAGE = "URL must not target private or link-local addresses"

BLOCKED_URLS = [
    "http://192.0.2.1/docs",
    "http://198.51.100.1/",
    "http://203.0.113.1/path",
    "http://198.18.0.1/benchmark",
    "http://[2001:db8::1]/docs",
    "http://[::]/",
]

ALLOWED_URLS = [
    "http://[ff02::1]/neighbors",
    "http://[fec0::1]/site",
    "http://192.88.99.1/relay",
    "http://[64:ff9b::8.8.8.8]/nat64",
    "http://0x7f000001/admin",
    "http://2130706433/admin",
]


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


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_ssrf_blocks_documentation_benchmark_and_ipv6_unspecified(url: str) -> None:
    ok, err = _validate_url_block_ssrf(url)
    assert ok is False
    assert err == PRIVATE_MESSAGE


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_ssrf_allows_multicast_sitelocal_nat64_and_nonliteral_hosts(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (True, None)


@pytest.mark.parametrize(
    "url",
    ["http://192.0.2.1/docs", "http://[2001:db8::1]/docs"],
)
def test_crawl_and_ingest_reject_documentation_ranges_before_fetch(
    client, mock_clients, url: str
) -> None:
    with patch("core.detect_content_type", side_effect=AssertionError("HEAD must not run")):
        crawl = client.post("/crawl", json={"url": url, "depth": 1, "breadth": 1})
        ingest = client.post(
            "/ingest",
            json={
                "url": url,
                "database": "test_db",
                "collection": "docs",
                "depth": 1,
                "breadth": 1,
            },
        )

    assert crawl.status_code == 400
    assert crawl.get_json()["error"] == PRIVATE_MESSAGE
    assert ingest.status_code == 400
    assert ingest.get_json()["error"] == PRIVATE_MESSAGE
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_crawl_reaches_browser_for_decimal_hostname(client, mock_clients) -> None:
    """A decimal hostname is not an IP literal, so the guard does not block it."""
    pages = [{"url": "http://2130706433/admin", "title": "Admin", "markdown": "secret"}]
    with patch("core.playwright_crawl", return_value=pages) as playwright:
        response = client.post("/crawl", json={"url": "http://2130706433/admin"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["crawl_method"] == "playwright"
    assert payload["combined_text"] == "Admin | Source: http://2130706433/admin\nsecret"
    playwright.assert_called_once_with("http://2130706433/admin", 2, 10)
    mock_clients["firecrawl"].crawl.assert_not_called()
