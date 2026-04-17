"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet
from flask import Flask, jsonify

import async_utils
import audit_logger
import security
from audit_logger import AuditLogger
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


def test_init_app_api_key_trims_and_disables_when_blank(monkeypatch: Any) -> None:
    monkeypatch.setenv("APP_API_KEY", "  top-secret  ")
    security.init_app_api_key()
    assert security._app_api_key == "top-secret"

    monkeypatch.setenv("APP_API_KEY", "   ")
    security.init_app_api_key()
    assert security._app_api_key is None


def test_require_api_key_decorator_enforces_header(monkeypatch: Any) -> None:
    monkeypatch.setattr(security, "_app_api_key", "expected-key")

    app = Flask(__name__)

    @app.get("/secured")
    @security.require_api_key
    def secured():
        return jsonify({"ok": True}), 200

    client = app.test_client()
    missing = client.get("/secured")
    assert missing.status_code == 401
    assert missing.get_json() == {"error": "Missing API key."}

    wrong = client.get("/secured", headers={"x-api-key": "wrong"})
    assert wrong.status_code == 403
    assert wrong.get_json() == {"error": "Invalid API key."}

    allowed = client.get("/secured", headers={"x-api-key": "expected-key"})
    assert allowed.status_code == 200
    assert allowed.get_json() == {"ok": True}


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_require_api_key_async_decorator_enforces_header(
    monkeypatch: Any, anyio_backend: str
) -> None:
    monkeypatch.setattr(security, "_app_api_key", "expected-key")
    app = Flask(__name__)

    @security.require_api_key_async
    async def secured_async():
        return jsonify({"ok": True}), 200

    with app.test_request_context("/", headers={}):
        missing_response, missing_status = await secured_async()
        assert missing_status == 401
        assert missing_response.get_json() == {"error": "Missing API key."}

    with app.test_request_context("/", headers={"x-api-key": "wrong"}):
        denied_response, denied_status = await secured_async()
        assert denied_status == 403
        assert denied_response.get_json() == {"error": "Invalid API key."}

    with app.test_request_context("/", headers={"x-api-key": "expected-key"}):
        ok_response, ok_status = await secured_async()
        assert ok_status == 200
        assert ok_response.get_json() == {"ok": True}


def test_load_config_auto_enables_auth_when_api_key_present(monkeypatch: Any) -> None:
    monkeypatch.setenv("COMPLIANCE_API_KEY", "compliance-secret")
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)
    config = load_config()
    assert config.api_key == "compliance-secret"
    assert config.auth_required is True


def test_load_config_auth_required_env_overrides_auto_enable(monkeypatch: Any) -> None:
    monkeypatch.setenv("COMPLIANCE_API_KEY", "compliance-secret")
    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "0")
    config = load_config()
    assert config.auth_required is False


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_run_in_thread_uses_asyncio_to_thread(anyio_backend: str, monkeypatch: Any) -> None:
    calls: list[tuple[Any, ...]] = []

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(async_utils.asyncio, "to_thread", fake_to_thread)
    result = await async_utils.run_in_thread(lambda a, b: a + b, 2, 3)
    assert result == 5
    assert calls == [((2, 3), {})]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_run_in_thread_falls_back_to_event_loop(
    anyio_backend: str, monkeypatch: Any
) -> None:
    monkeypatch.delattr(async_utils.asyncio, "to_thread", raising=False)

    class DummyLoop:
        def __init__(self) -> None:
            self.executor = object()

        async def run_in_executor(self, executor: Any, callback: Any) -> Any:
            self.executor = executor
            return callback()

    loop = DummyLoop()
    monkeypatch.setattr(async_utils.asyncio, "get_event_loop", lambda: loop)
    result = await async_utils.run_in_thread(lambda value: value * 2, 4)
    assert result == 8
    assert loop.executor is None


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_encrypts_record_and_raw_payload(
    anyio_backend: str, monkeypatch: Any
) -> None:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("AUDIT_LOG_KEY", key)

    async def inline_run(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(audit_logger, "run_in_thread", inline_run)
    fixed_now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(audit_logger, "utc_now", lambda: fixed_now)

    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True
    config.audit_collection = "audit"

    mock_collection = MagicMock()
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_collection
    logger = AuditLogger(mock_mongo, config)

    sections = [{"section_text": "Collection and use details"}]
    await logger.log(
        database="privacy-compliance",
        policy_id="policy-1",
        jurisdiction="CA",
        policy_text="Raw policy text",
        sections=sections,
        summary={"status": "ok"},
    )

    inserted = mock_collection.insert_one.call_args[0][0]
    assert inserted["policy_id"] == "policy-1"
    assert inserted["jurisdiction"] == "CA"
    assert inserted["policy_hash"] != "Raw policy text"
    assert "encrypted_record" in inserted
    assert "encrypted_payload" in inserted
    assert "policy_text" not in inserted

    fernet = Fernet(key.encode("utf-8"))
    encrypted_record = json.loads(
        fernet.decrypt(inserted["encrypted_record"].encode("utf-8")).decode("utf-8")
    )
    assert encrypted_record["policy_hash"] == inserted["policy_hash"]
    assert encrypted_record["created_at"] == fixed_now.isoformat()

    encrypted_payload = json.loads(
        fernet.decrypt(inserted["encrypted_payload"].encode("utf-8")).decode("utf-8")
    )
    assert encrypted_payload["policy_text"] == "Raw policy text"
    assert encrypted_payload["sections"] == sections


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_without_key_skips_encryption(
    anyio_backend: str, monkeypatch: Any
) -> None:
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)

    async def inline_run(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(audit_logger, "run_in_thread", inline_run)
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True
    config.audit_collection = "audit"

    mock_collection = MagicMock()
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_collection
    logger = AuditLogger(mock_mongo, config)

    await logger.log(
        database="privacy-compliance",
        policy_id=None,
        jurisdiction="US",
        policy_text="Raw policy text",
        sections=[{"section_text": "text"}],
        summary={},
    )

    inserted = mock_collection.insert_one.call_args[0][0]
    assert "encrypted_record" not in inserted
    assert "encrypted_payload" not in inserted
