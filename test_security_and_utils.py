"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

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


def test_authorize_request_skips_checks_when_auth_disabled() -> None:
    config = load_config()
    config.auth_required = False
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    request = SimpleNamespace(headers={})
    authorize_request(config, request)


def test_authorize_request_strips_provided_key_but_not_configured_key() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = []
    authorize_request(config, SimpleNamespace(headers={"x-api-key": "  secret  "}))

    config.api_key = " secret "
    request = SimpleNamespace(headers={"x-api-key": "secret"})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 403
    assert exc.value.message == "Invalid API key."


def test_authorize_request_whitespace_key_is_missing() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = []
    request = SimpleNamespace(headers={"x-api-key": "   "})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 401
    assert exc.value.message == "Missing API key."


def test_authorize_request_empty_configured_key_still_enforces_role() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = ""
    config.allowed_roles = ["admin"]
    authorize_request(
        config,
        SimpleNamespace(headers={"x-api-key": "any-caller-key", "x-role": "Admin"}),
    )
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(
            config,
            SimpleNamespace(headers={"x-api-key": "any-caller-key", "x-role": "viewer"}),
        )
    assert exc.value.status_code == 403
    assert exc.value.message == "Insufficient role."


def test_authorize_request_uses_configured_role_header_and_strips_role() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["Compliance"]
    config.default_role_header = "x-user-role"
    authorize_request(
        config,
        SimpleNamespace(headers={"x-api-key": "secret", "x-role": "admin", "x-user-role": "  compliance  "}),
    )
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(
            config,
            SimpleNamespace(headers={"x-api-key": "secret", "x-role": "compliance"}),
        )
    assert exc.value.status_code == 403
    assert exc.value.message == "Insufficient role."


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
