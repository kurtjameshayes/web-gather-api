"""SSRF address-class edges for ingest and crawl.

Existing coverage rejects literal localhost, RFC1918 samples, ::1, and
link-local metadata. These cases pin the classifier edges those tests do
not: unspecified and broadcast addresses, IPv4-mapped IPv6, userinfo, IPv6
unique-local, and the ranges Python does not treat as private.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()

from core import _validate_url_block_ssrf, core_bp, init_core


@pytest.mark.parametrize(
    ("url", "allowed", "message_part"),
    [
        ("HTTP://example.com/policy", True, None),
        ("http://0.0.0.0/", False, "private"),
        ("http://255.255.255.255/", False, "private"),
        ("http://172.20.5.5/internal", False, "private"),
        ("http://[fc00::1]/", False, "private"),
        ("http://[::ffff:127.0.0.1]/", False, "private"),
        ("http://[::ffff:10.1.2.3]/secret", False, "private"),
        ("http://user:pass@127.0.0.1/admin", False, "private"),
        ("http://[::ffff:8.8.8.8]/", True, None),
        ("http://100.64.0.1/", True, None),
        ("http://127.1/", True, None),
        ("http://224.0.0.1/", True, None),
    ],
)
def test_ssrf_address_classes(url: str, allowed: bool, message_part: str | None) -> None:
    """Mapped loopback is private but not is_loopback; shared space and short IPv4 are allowed."""
    ok, message = _validate_url_block_ssrf(url)
    assert ok is allowed
    if allowed:
        assert message is None
    else:
        assert message is not None
        assert message_part in message.lower()
        assert "localhost" not in message.lower()


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
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
        "http://[::ffff:127.0.0.1]/admin",
        "http://0.0.0.0/",
        "http://user:pass@10.1.2.3/",
    ],
)
def test_ingest_rejects_mapped_and_unspecified_before_fetch(
    client, mock_clients, url: str
) -> None:
    response = client.post(
        "/ingest",
        json={"url": url, "database": "test_db", "collection": "docs"},
    )
    assert response.status_code == 400
    assert "private" in response.get_json()["error"].lower()
    mock_clients["firecrawl"].crawl.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:127.0.0.1]/admin",
        "http://[fc00::1]/",
    ],
)
def test_crawl_rejects_mapped_and_ula_before_fetch(
    client, mock_clients, url: str
) -> None:
    response = client.post("/crawl", json={"url": url})
    assert response.status_code == 400
    assert "private" in response.get_json()["error"].lower()
    mock_clients["firecrawl"].crawl.assert_not_called()
