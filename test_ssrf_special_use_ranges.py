"""SSRF checks for special-use ranges not covered by earlier address-class tests.

Python's ``ipaddress`` marks these as private, so ``_validate_url_block_ssrf``
rejects them. Nearby addresses that are global stay allowed.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import _validate_url_block_ssrf, core_bp, init_core


PRIVATE = "URL must not target private or link-local addresses"

BLOCKED_URLS = [
    "http://192.0.0.1/ietf",
    "http://192.0.0.8/dummy",
    "http://user:pass@240.0.0.1/class-e",
    "http://255.255.255.254/reserved",
    "http://[2001:0::1]/teredo",
    "http://[100::1]/discard",
    "http://[64:ff9b:1::8.8.8.8]/local-nat64",
    "http://[2002::1]/6to4",
]

ALLOWED_URLS = [
    "http://192.0.0.9/public",
    "http://198.20.0.1/outside-benchmark",
    "http://172.32.0.1/outside-rfc1918",
    "http://[3fff::1]/global",
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
def test_special_use_ranges_are_private(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (False, PRIVATE)


@pytest.mark.parametrize("url", ALLOWED_URLS)
def test_adjacent_global_addresses_are_allowed(url: str) -> None:
    assert _validate_url_block_ssrf(url) == (True, None)


def test_crawl_blocks_class_e_before_fetch(client, mock_clients) -> None:
    with patch("core.playwright_crawl") as playwright, patch(
        "core.detect_content_type"
    ) as detect:
        response = client.post("/crawl", json={"url": "http://240.0.0.1/class-e"})
    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    playwright.assert_not_called()
    detect.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_blocks_teredo_before_fetch(client, mock_clients) -> None:
    with patch("core.detect_content_type") as detect, patch("core.requests.head") as head:
        response = client.post(
            "/ingest",
            json={
                "url": "http://[2001:0::1]/teredo",
                "database": "test_db",
                "collection": "docs",
            },
        )
    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    detect.assert_not_called()
    head.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_crawl_blocks_6to4_prefix(client, mock_clients) -> None:
    with patch("core.playwright_crawl") as playwright:
        response = client.post("/crawl", json={"url": "http://[2002::1]/6to4"})
    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    playwright.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_ingest_blocks_local_nat64(client, mock_clients) -> None:
    with patch("core.detect_content_type") as detect:
        response = client.post(
            "/ingest",
            json={
                "url": "http://[64:ff9b:1::8.8.8.8]/local-nat64",
                "database": "test_db",
                "collection": "docs",
                "mode": "append",
            },
        )
    assert response.status_code == 400
    assert response.get_json()["error"] == PRIVATE
    detect.assert_not_called()
    mock_clients["firecrawl"].crawl.assert_not_called()


def test_crawl_allows_address_just_outside_ietf_block(client, mock_clients) -> None:
    pages = [{"url": "http://192.0.0.9/public", "title": "Public", "markdown": "visible"}]
    with patch("core.playwright_crawl", return_value=pages) as playwright:
        response = client.post(
            "/crawl",
            json={"url": "http://192.0.0.9/public", "depth": 2, "breadth": 10},
        )
    assert response.status_code == 200
    body = response.get_json()
    assert body["crawl_method"] == "playwright"
    assert body["combined_text"] == "Public | Source: http://192.0.0.9/public\nvisible"
    playwright.assert_called_once_with("http://192.0.0.9/public", 2, 10)
    mock_clients["firecrawl"].crawl.assert_not_called()
