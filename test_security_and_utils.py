"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from async_utils import run_in_thread
from audit_logger import AuditLogger
from cache import SimpleLRUCache
from compliance_config import load_config
from compliance_utils import (
    clamp,
    extract_json_block,
    hash_text,
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
async def test_run_in_thread_falls_back_to_executor(
    monkeypatch: Any, anyio_backend: str
) -> None:
    monkeypatch.delattr(asyncio, "to_thread", raising=False)

    result = await run_in_thread(lambda prefix, value: f"{prefix}-{value}", "case", value=7)

    assert result == "case-7"


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


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_disabled_skips_storage(
    monkeypatch: Any, anyio_backend: str
) -> None:
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    config = load_config()
    config.enable_audit_logging = False
    mongo = MagicMock()

    logger = AuditLogger(mongo, config)
    await logger.log(
        database="privacy",
        policy_id="policy-1",
        jurisdiction="CA",
        policy_text="raw policy text",
        sections=[{"section_text": "raw section text"}],
        summary={"status": "ok"},
    )

    mongo.__getitem__.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_stores_hash_only_record_without_key(
    monkeypatch: Any, anyio_backend: str
) -> None:
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = False
    collection = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    logger = AuditLogger(mongo, config)
    await logger.log(
        database="privacy",
        policy_id="policy-1",
        jurisdiction="CA",
        policy_text="raw policy text",
        sections=[{"section_text": "raw section text"}],
        summary={"status": "ok"},
    )

    collection.insert_one.assert_called_once()
    record = collection.insert_one.call_args.args[0]
    assert record["policy_hash"] == hash_text("raw policy text")
    assert record["section_hashes"] == [hash_text("raw section text")]
    assert "encrypted_record" not in record
    assert "encrypted_payload" not in record
    assert "raw policy text" not in json.dumps(record, default=str)
    assert "raw section text" not in json.dumps(record, default=str)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_encrypts_record_and_allowed_raw_payload(
    monkeypatch: Any, anyio_backend: str
) -> None:
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True
    collection = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    logger = AuditLogger(mongo, config)
    await logger.log(
        database="privacy",
        policy_id="policy-1",
        jurisdiction="CA",
        policy_text="raw policy text",
        sections=[{"section_text": "raw section text"}],
        summary={"status": "ok"},
    )

    record = collection.insert_one.call_args.args[0]
    fernet = Fernet(key)
    encrypted_record = json.loads(
        fernet.decrypt(record["encrypted_record"].encode("utf-8")).decode("utf-8")
    )
    encrypted_payload = json.loads(
        fernet.decrypt(record["encrypted_payload"].encode("utf-8")).decode("utf-8")
    )

    assert encrypted_record["policy_hash"] == hash_text("raw policy text")
    assert "policy_text" not in encrypted_record
    assert encrypted_payload == {
        "policy_text": "raw policy text",
        "sections": [{"section_text": "raw section text"}],
    }
    assert "raw policy text" not in json.dumps(
        {key: value for key, value in record.items() if key != "encrypted_payload"},
        default=str,
    )


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
