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
    validate_collection_name,
)
from rate_limiter import RateLimiter
from redactor import Redactor
from security import AuthorizationError, authorize_request
import security


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


def test_authorize_request_skips_when_auth_disabled() -> None:
    config = load_config()
    config.auth_required = False
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    authorize_request(config, SimpleNamespace(headers={}))


def test_authorize_request_strips_key_and_matches_role_case() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["Compliance"]
    config.default_role_header = "x-custom-role"
    request = SimpleNamespace(
        headers={"x-api-key": "  secret  ", "x-custom-role": " COMPLIANCE "}
    )
    authorize_request(config, request)


def test_authorize_request_empty_roles_is_key_only() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = []
    authorize_request(config, SimpleNamespace(headers={"x-api-key": "secret"}))


def test_authorize_request_blank_role_is_rejected() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    request = SimpleNamespace(headers={"x-api-key": "secret", "x-role": "   "})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 403


@pytest.fixture
def restored_app_api_key():
    previous = security._app_api_key
    yield
    security._app_api_key = previous


def _guarded_app() -> Flask:
    app = Flask(__name__)

    @app.get("/sync")
    @security.require_api_key
    def sync_route():
        return jsonify({"ok": True, "handler": "sync"})

    @app.get("/async")
    @security.require_api_key_async
    async def async_route():
        return jsonify({"ok": True, "handler": "async"})

    return app


def test_require_api_key_passthrough_when_unset(restored_app_api_key: None) -> None:
    security._app_api_key = None
    response = _guarded_app().test_client().get("/sync")
    assert response.status_code == 200
    assert response.get_json()["handler"] == "sync"


def test_require_api_key_missing_and_invalid(restored_app_api_key: None) -> None:
    security._app_api_key = "secret"
    client = _guarded_app().test_client()

    missing = client.get("/sync")
    assert missing.status_code == 401
    assert missing.get_json()["error"] == "Missing API key."

    blank = client.get("/sync", headers={"x-api-key": "   "})
    assert blank.status_code == 401

    wrong = client.get("/sync", headers={"x-api-key": "nope"})
    assert wrong.status_code == 403
    assert wrong.get_json()["error"] == "Invalid API key."


def test_require_api_key_accepts_padded_key(restored_app_api_key: None) -> None:
    security._app_api_key = "secret"
    response = _guarded_app().test_client().get("/sync", headers={"x-api-key": " secret "})
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_require_api_key_async_rejects_before_handler(restored_app_api_key: None) -> None:
    security._app_api_key = "secret"
    called = False

    async def handler():
        nonlocal called
        called = True
        return jsonify({"ok": True})

    wrapped = security.require_api_key_async(handler)
    app = Flask(__name__)
    with app.test_request_context("/async"):
        body, status = asyncio.run(wrapped())
    assert status == 401
    assert body.get_json()["error"] == "Missing API key."
    assert called is False

    with app.test_request_context("/async", headers={"x-api-key": "secret"}):
        response = asyncio.run(wrapped())
    assert called is True
    assert response.get_json()["ok"] is True


def test_require_api_key_async_passthrough_when_unset(restored_app_api_key: None) -> None:
    security._app_api_key = None

    async def handler():
        return {"ok": True}

    wrapped = security.require_api_key_async(handler)
    app = Flask(__name__)
    with app.test_request_context("/async"):
        assert asyncio.run(wrapped()) == {"ok": True}


def test_redactor_masks_common_pii() -> None:
    redactor = Redactor()
    result = redactor.redact("Email me at a@example.com or call 555-123-4567.")
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert result.redaction_count >= 2


def test_redactor_distinguishes_ssn_phone_and_card() -> None:
    redactor = Redactor()
    result = redactor.redact(
        "ssn 123-45-6789 phone (555) 123-4567 card 4111111111111111"
    )
    assert "123-45-6789" not in result.redacted_text
    assert "4111111111111111" not in result.redacted_text
    assert "(555) 123-4567" not in result.redacted_text
    assert "[REDACTED_SSN]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "[REDACTED_CARD]" in result.redacted_text
    assert result.redaction_count == 3


def test_redactor_empty_and_custom_patterns() -> None:
    redactor = Redactor()
    empty = redactor.redact("")
    assert empty.redacted_text == ""
    assert empty.redaction_count == 0
    blank = redactor.redact(None)  # type: ignore[arg-type]
    assert blank.redacted_text == ""
    assert blank.redaction_count == 0

    custom = Redactor(patterns=[(r"secret", "[X]")])
    result = custom.redact("secret a@example.com")
    assert result.redacted_text == "[X] a@example.com"
    assert result.redaction_count == 1


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
