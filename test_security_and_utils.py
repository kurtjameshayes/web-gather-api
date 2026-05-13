"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask, jsonify

from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import (
    clamp,
    extract_json_block,
    jurisdiction_filter_values,
    normalize_jurisdiction,
    safe_truncate,
    slugify,
    truncate_at_sentence,
    validate_collection_name,
)
from rate_limiter import RateLimiter
from redactor import Redactor
from security import (
    AuthorizationError,
    authorize_request,
    init_app_api_key,
    require_api_key,
    require_api_key_async,
)


@pytest.fixture(autouse=True)
def reset_app_api_key(monkeypatch: Any):
    """Keep APP_API_KEY decorator state isolated between tests."""
    monkeypatch.delenv("APP_API_KEY", raising=False)
    init_app_api_key()
    yield
    monkeypatch.delenv("APP_API_KEY", raising=False)
    init_app_api_key()


def test_authorize_request_missing_key() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    request = SimpleNamespace(headers={})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 401


def test_authorize_request_invalid_key() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    request = SimpleNamespace(headers={"x-api-key": "wrong"})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 403


def test_authorize_request_role_blocked() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    request = SimpleNamespace(headers={"x-api-key": "secret", "x-role": "viewer"})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 403


def test_authorize_request_role_allowed() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    request = SimpleNamespace(headers={"x-api-key": "secret", "x-role": "admin"})
    authorize_request(config, request)


def test_authorize_request_uses_custom_role_header() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    config.default_role_header = "x-tenant-role"

    request = SimpleNamespace(
        headers={"x-api-key": "secret", "x-tenant-role": "Admin"}
    )
    authorize_request(config, request)

    default_header_request = SimpleNamespace(
        headers={"x-api-key": "secret", "x-role": "admin"}
    )
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, default_header_request)
    assert exc.value.status_code == 403


def test_require_api_key_allows_when_unconfigured() -> None:
    app = Flask(__name__)

    @require_api_key
    def protected():
        return jsonify({"ok": True})

    with app.test_request_context("/"):
        response = protected()

    assert response.status_code == 200
    assert response.get_json() == {"ok": True}


def test_require_api_key_sync_enforces_configured_app_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("APP_API_KEY", "app-secret")
    init_app_api_key()
    app = Flask(__name__)

    @require_api_key
    def protected():
        return jsonify({"ok": True})

    with app.test_request_context("/"):
        response, status = protected()
    assert status == 401
    assert response.get_json() == {"error": "Missing API key."}

    with app.test_request_context("/", headers={"x-api-key": "wrong"}):
        response, status = protected()
    assert status == 403
    assert response.get_json() == {"error": "Invalid API key."}

    with app.test_request_context("/", headers={"x-api-key": "app-secret"}):
        response = protected()
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}


def test_require_api_key_async_enforces_configured_app_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("APP_API_KEY", "app-secret")
    init_app_api_key()
    app = Flask(__name__)

    @require_api_key_async
    async def protected():
        return jsonify({"ok": True})

    with app.test_request_context("/"):
        response, status = asyncio.run(protected())
    assert status == 401
    assert response.get_json() == {"error": "Missing API key."}

    with app.test_request_context("/", headers={"x-api-key": "wrong"}):
        response, status = asyncio.run(protected())
    assert status == 403
    assert response.get_json() == {"error": "Invalid API key."}

    with app.test_request_context("/", headers={"x-api-key": "app-secret"}):
        response = asyncio.run(protected())
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}


def test_redactor_masks_common_pii() -> None:
    redactor = Redactor()
    result = redactor.redact("Email me at a@example.com or call 555-123-4567.")
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert result.redaction_count >= 2


def test_cache_eviction_and_expiry(monkeypatch: Any) -> None:
    now = 1000.0

    def fake_time() -> float:
        return now

    monkeypatch.setattr(time, "time", fake_time)
    cache = SimpleLRUCache(max_size=1, ttl_seconds=10)
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.get("a") is None
    assert cache.get("b") == "2"

    now = 2000.0
    assert cache.get("b") is None


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_rate_limiter_blocks_then_allows(monkeypatch: Any, anyio_backend: str) -> None:
    now = 1000.0

    def fake_time() -> float:
        return now

    monkeypatch.setattr(time, "monotonic", fake_time)
    limiter = RateLimiter(rate_per_minute=1)
    assert await limiter.allow() is True
    assert await limiter.allow() is False
    now = 1060.0
    assert await limiter.allow() is True


def test_compliance_utils_helpers() -> None:
    assert normalize_jurisdiction("United States") == "US"
    ca_values = jurisdiction_filter_values("CA")
    assert "CA" in ca_values
    assert "California" in ca_values or "california" in ca_values
    assert normalize_jurisdiction("California") == "CA"
    assert clamp(1.5) == 1.0
    assert safe_truncate("abcdef", 4) == "a..."
    assert slugify("Data Retention") == "data_retention"
    assert extract_json_block("x {\"a\":1} y") == '{"a":1}'
    assert validate_collection_name("valid_name-1") is True
    assert validate_collection_name("bad name") is False


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("Alpha. Beta continues beyond limit", 12, "Alpha."),
        ("Alpha! Beta continues beyond limit", 12, "Alpha!"),
        ("Alpha? Beta continues beyond limit", 12, "Alpha?"),
        ("No boundary here   and more", 16, "No boundary here"),
        ("Short text", 50, "Short text"),
        ("", 10, ""),
        ("Keep original", 0, "Keep original"),
    ],
)
def test_truncate_at_sentence_edges(text: str, max_chars: int, expected: str) -> None:
    assert truncate_at_sentence(text, max_chars) == expected
