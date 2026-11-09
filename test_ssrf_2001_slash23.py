"""SSRF decisions for the IPv6 exceptions inside 2001::/23.

On Python 3.12, ``ipaddress`` marks 2001::/23 private except a few IETF
assignments. Earlier coverage pinned Teredo ``2001::1``, benchmark
``2001:2::1``, ORCHIDv1 ``2001:10::1``, and ORCHIDv2 ``2001:20::1``. It did
not pin the single-address anycast holes, the AMT and AS112-v6 exceptions, or
the addresses on either side of those holes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core import WEB_GATHER_DB, _validate_url_block_ssrf


PRIVATE_MESSAGE = "URL must not target private or link-local addresses"

BLOCKED_URLS = [
    "http://[2001:1::]/",
    "http://[2001:1::3]/",
    "http://[2001:0001:0000:0000:0000:0000:0000:0003]/",
    "http://user:pass@[2001:1::3]/secret",
    "http://[2001:1::3]:8080/latest",
    "http://[2001:2:ffff:ffff:ffff:ffff:ffff:ffff]/",
    "http://[2001:4:111:ffff:ffff:ffff:ffff:ffff]/",
    "http://[2001:4:113::1]/",
    "http://[2001:40::1]/",
    "http://[2001:1ff:ffff:ffff:ffff:ffff:ffff:ffff]/",
]

ALLOWED_URLS = [
    "http://[2001:1::1]/",
    "http://[2001:1::2]/",
    "http://[2001:0001:0000:0000:0000:0000:0000:0001]/",
    "http://[2001:3::1]/",
    "http://[2001:3:ffff:ffff:ffff:ffff:ffff:ffff]/",
    "http://[2001:4:112::1]/",
    "http://[2001:4:112:ffff:ffff:ffff:ffff:ffff]/",
    "http://[2001:3f:ffff:ffff:ffff:ffff:ffff:ffff]/",
    "http://[2001:200::1]/",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_helper_blocks_2001_slash23_outside_the_exceptions(url: str) -> None:
    ok, err = _validate_url_block_ssrf(url)
    assert ok is False
    assert err == PRIVATE_MESSAGE


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_helper_allows_2001_slash23_exceptions_and_the_next_block(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (True, None)


@pytest.mark.parametrize(
    "url",
    [
        "http://[2001:1::3]/",
        "http://user:pass@[2001:1::3]/secret",
        "http://[2001:1ff:ffff:ffff:ffff:ffff:ffff:ffff]/",
    ],
)
def test_crawl_blocks_before_browsers(client_core, mock_clients_core, url: str) -> None:
    with patch("core.playwright_crawl") as playwright, patch(
        "core.puppeteer_crawl"
    ) as puppeteer, patch("core.selenium_crawl") as selenium:
        response = client_core.post("/crawl", json={"url": url})

    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE_MESSAGE
    playwright.assert_not_called()
    puppeteer.assert_not_called()
    selenium.assert_not_called()
    mock_clients_core["firecrawl"].crawl.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://[2001:40::1]/",
        "http://[2001:0001:0000:0000:0000:0000:0000:0003]/",
    ],
)
def test_ingest_blocks_before_head_or_crawl(client_core, mock_clients_core, url: str) -> None:
    with patch("core.detect_content_type") as detect, patch("core.requests.head") as head:
        response = client_core.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE_MESSAGE
    detect.assert_not_called()
    head.assert_not_called()
    mock_clients_core["firecrawl"].crawl.assert_not_called()


def test_crawl_of_pcp_anycast_uses_playwright_defaults(client_core, mock_clients_core) -> None:
    url = "http://[2001:1::1]/"
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": url, "title": "T", "markdown": "Body"}],
    ) as playwright, patch("core.puppeteer_crawl") as puppeteer:
        response = client_core.post("/crawl", json={"url": url})

    assert response.status_code == 200
    assert response.get_json()["crawl_method"] == "playwright"
    playwright.assert_called_once_with(url, 2, 10)
    puppeteer.assert_not_called()
    mock_clients_core["firecrawl"].crawl.assert_not_called()


def test_crawl_of_as112_exception_uses_playwright_defaults(client_core, mock_clients_core) -> None:
    url = "http://[2001:4:112::1]/"
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": url, "title": "T", "markdown": "Body"}],
    ) as playwright, patch("core.puppeteer_crawl") as puppeteer:
        response = client_core.post("/crawl", json={"url": url})

    assert response.status_code == 200
    playwright.assert_called_once_with(url, 2, 10)
    puppeteer.assert_not_called()
    mock_clients_core["firecrawl"].crawl.assert_not_called()


def test_ingest_of_turn_anycast_reaches_head_and_firecrawl(client_core, mock_clients_core) -> None:
    url = "http://[2001:1::2]/"
    _wire_mongo(mock_clients_core["mongo"])
    mock_clients_core["firecrawl"].crawl.return_value = [
        {"url": url, "title": "T", "markdown": "Body"}
    ]
    with patch("core.detect_content_type", return_value="") as detect:
        response = client_core.post(
            "/ingest",
            json={"url": url, "database": "test_db", "collection": "docs"},
        )

    assert response.status_code == 200
    assert response.get_json()["document_type"] == "web"
    detect.assert_called_once_with(url)
    mock_clients_core["firecrawl"].crawl.assert_called_once_with(url=url, limit=5)


def _wire_mongo(mock_mongo) -> None:
    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str):
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
