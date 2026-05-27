"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

import security
from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import (
    clamp,
    extract_json_block,
    jurisdiction_filter_values,
    normalize_jurisdiction,
    safe_truncate,
    slugify,
    validate_collection_name,
)
from rate_limiter import RateLimiter
from redactor import Redactor
from security import AuthorizationError, authorize_request


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


def test_require_api_key_blocks_missing_header(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key
    def handler():
        return {"ok": True}

    with app.test_request_context("/"):
        response, status = handler()

    assert status == 401
    assert response.get_json()["error"] == "Missing API key."


def test_require_api_key_rejects_invalid_header(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key
    def handler():
        return {"ok": True}

    with app.test_request_context("/", headers={"x-api-key": "wrong"}):
        response, status = handler()

    assert status == 403
    assert response.get_json()["error"] == "Invalid API key."


def test_require_api_key_allows_valid_header(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key
    def handler():
        return {"ok": True}

    with app.test_request_context("/", headers={"x-api-key": "secret"}):
        result = handler()

    assert result == {"ok": True}


def test_require_api_key_async_blocks_missing_header(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key_async
    async def handler():
        return {"ok": True}

    with app.test_request_context("/"):
        response, status = asyncio.run(handler())

    assert status == 401
    assert response.get_json()["error"] == "Missing API key."


def test_require_api_key_async_allows_unconfigured_key(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", None)

    @security.require_api_key_async
    async def handler():
        return {"ok": True}

    with app.test_request_context("/"):
        result = asyncio.run(handler())

    assert result == {"ok": True}


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
