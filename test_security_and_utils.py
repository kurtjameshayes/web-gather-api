"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

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


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_stores_hashes_without_raw_policy_text(monkeypatch: Any, anyio_backend: str) -> None:
    """Audit logging defaults to hashed policy/section data only."""
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = False
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    logger = AuditLogger(mock_mongo, config)

    await logger.log(
        database="privacy-compliance",
        policy_id="pol-1",
        jurisdiction="CA",
        policy_text="Sensitive policy text with user@example.com",
        sections=[{"section_text": "Consumer deletion rights."}],
        summary={"missing": 0},
    )

    inserted = mock_coll.insert_one.call_args[0][0]
    serialized = repr(inserted)
    assert "Sensitive policy text" not in serialized
    assert "Consumer deletion rights" not in serialized
    assert "user@example.com" not in serialized
    assert inserted["policy_hash"]
    assert inserted["section_hashes"]
    assert "encrypted_payload" not in inserted


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_audit_logger_encrypts_raw_payload_when_allowed(monkeypatch: Any, anyio_backend: str) -> None:
    """Raw audit payloads are only persisted inside decryptable encrypted fields."""
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    logger = AuditLogger(mock_mongo, config)

    await logger.log(
        database="privacy-compliance",
        policy_id="pol-1",
        jurisdiction="CA",
        policy_text="Raw policy text requiring encryption.",
        sections=[{"section_text": "Raw section text requiring encryption."}],
        summary={"addressed": 1},
    )

    inserted = mock_coll.insert_one.call_args[0][0]
    assert "policy_text" not in inserted
    assert "sections" not in inserted
    assert "encrypted_record" in inserted
    assert "encrypted_payload" in inserted

    fernet = Fernet(key)
    payload = json.loads(
        fernet.decrypt(inserted["encrypted_payload"].encode("utf-8")).decode("utf-8")
    )
    assert payload["policy_text"] == "Raw policy text requiring encryption."
    assert payload["sections"] == [{"section_text": "Raw section text requiring encryption."}]
