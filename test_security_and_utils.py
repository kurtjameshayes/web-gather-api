"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

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
import security
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


def test_load_config_auto_requires_auth_when_api_key_configured(monkeypatch: Any) -> None:
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", "/tmp/web-gather-missing-config.json")
    monkeypatch.setenv("COMPLIANCE_API_KEY", "secret")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)

    config = load_config()

    assert config.api_key == "secret"
    assert config.auth_required is True


def test_load_config_honors_explicit_auth_required_false(monkeypatch: Any) -> None:
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", "/tmp/web-gather-missing-config.json")
    monkeypatch.setenv("COMPLIANCE_API_KEY", "secret")
    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "false")
    monkeypatch.delenv("API_KEY", raising=False)

    config = load_config()

    assert config.api_key == "secret"
    assert config.auth_required is False


def test_require_api_key_allows_unconfigured_key(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", None)

    @security.require_api_key
    def handler() -> str:
        return "ok"

    with app.test_request_context("/"):
        assert handler() == "ok"


@pytest.mark.parametrize(
    ("headers", "expected_status", "expected_error"),
    [
        ({}, 401, "Missing API key."),
        ({"x-api-key": "wrong"}, 403, "Invalid API key."),
    ],
)
def test_require_api_key_rejects_missing_or_invalid_key(
    monkeypatch: Any,
    headers: dict[str, str],
    expected_status: int,
    expected_error: str,
) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key
    def handler() -> str:
        return "ok"

    with app.test_request_context("/", headers=headers):
        response, status = handler()

    assert status == expected_status
    assert response.get_json() == {"error": expected_error}


def test_require_api_key_accepts_valid_key(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key
    def handler() -> str:
        return "ok"

    with app.test_request_context("/", headers={"x-api-key": "secret"}):
        assert handler() == "ok"


def test_require_api_key_async_accepts_valid_key(monkeypatch: Any) -> None:
    app = Flask(__name__)
    monkeypatch.setattr(security, "_app_api_key", "secret")

    @security.require_api_key_async
    async def handler() -> str:
        return "ok"

    with app.test_request_context("/", headers={"x-api-key": "secret"}):
        assert asyncio.run(handler()) == "ok"


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
