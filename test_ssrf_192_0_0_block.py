"""SSRF decisions for the rest of 192.0.0.0/24.

On Python 3.12, ``ipaddress`` marks 192.0.0.0/24 private except 192.0.0.9 and
192.0.0.10. Earlier coverage pinned .1, .8, .9, .10, .170, and .171 only.
Addresses after the hole, the end of the /24, and the same hosts written as
IPv4-mapped IPv6 stay on the private side of ``_validate_url_block_ssrf``.
192.0.1.1 is the next block and is allowed. A trailing dot is not parsed as an
IP literal, so the guard does not block it.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core import _validate_url_block_ssrf


PRIVATE_MESSAGE = "URL must not target private or link-local addresses"

BLOCKED_URLS = [
    "http://192.0.0.11/",
    "http://192.0.0.172/",
    "http://192.0.0.255/",
    "http://user:pass@192.0.0.11/secret",
    "http://192.0.0.11:8080/latest",
    "http://[::ffff:192.0.0.11]/",
    "http://[::ffff:192.0.0.172]/",
    "http://[::ffff:192.0.0.255]/",
]

ALLOWED_URLS = [
    "http://192.0.1.1/",
    "http://[::ffff:192.0.0.10]/",
    "http://[::ffff:192.0.1.1]/",
    "http://192.0.0.11./",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_helper_blocks_remainder_of_192_0_0_slash24(url: str) -> None:
    ok, err = _validate_url_block_ssrf(url)
    assert ok is False
    assert err == PRIVATE_MESSAGE


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_helper_allows_neighbor_mapped_hole_and_trailing_dot(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (True, None)


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.0.11/",
        "http://[::ffff:192.0.0.255]/",
        "http://user:pass@192.0.0.172/secret",
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
        "http://192.0.0.255/",
        "http://[::ffff:192.0.0.11]/",
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


def test_crawl_of_192_0_1_1_uses_playwright_defaults(client_core, mock_clients_core) -> None:
    url = "http://192.0.1.1/"
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


def test_crawl_of_mapped_hole_uses_playwright_defaults(client_core, mock_clients_core) -> None:
    url = "http://[::ffff:192.0.0.10]/"
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": url, "title": "T", "markdown": "Body"}],
    ) as playwright, patch("core.puppeteer_crawl") as puppeteer:
        response = client_core.post("/crawl", json={"url": url})

    assert response.status_code == 200
    playwright.assert_called_once_with(url, 2, 10)
    puppeteer.assert_not_called()
    mock_clients_core["firecrawl"].crawl.assert_not_called()


def test_crawl_trailing_dot_is_not_an_ip_literal(client_core, mock_clients_core) -> None:
    url = "http://192.0.0.11./"
    with patch(
        "core.playwright_crawl",
        return_value=[{"url": url, "title": "T", "markdown": "Body"}],
    ) as playwright:
        response = client_core.post("/crawl", json={"url": url})

    assert response.status_code == 200
    playwright.assert_called_once_with(url, 2, 10)
    mock_clients_core["firecrawl"].crawl.assert_not_called()


def test_ingest_of_mapped_hole_reaches_head_and_firecrawl(client_core, mock_clients_core) -> None:
    url = "http://[::ffff:192.0.0.10]/"
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
    from unittest.mock import MagicMock

    from core import WEB_GATHER_DB

    db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str):
        return wg_db if name == WEB_GATHER_DB else db

    mock_mongo.__getitem__.side_effect = _get_db
    db.__getitem__.return_value = MagicMock()
    wg_db.__getitem__.return_value = MagicMock()
