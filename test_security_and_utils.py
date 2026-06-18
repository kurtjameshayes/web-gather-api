"""Tests for security, utilities, cache, and redaction helpers."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

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


@pytest.fixture(autouse=True)
def reset_app_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(security, "_app_api_key", None)


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


def test_app_api_key_enforced_across_core_db_and_util_blueprints(monkeypatch: pytest.MonkeyPatch) -> None:
    """APP_API_KEY protects each non-compliance blueprint before handlers run."""
    from core import core_bp
    from db import db_bp
    from util import init_util, util_bp

    monkeypatch.setattr(security, "_app_api_key", "secret")

    wg_db = MagicMock()
    wg_db.__getitem__.return_value.find.return_value = []
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = Flask(__name__)
    app.register_blueprint(core_bp)
    app.register_blueprint(db_bp)
    app.register_blueprint(util_bp)
    client = app.test_client()

    protected_routes = [
        ("post", "/gather", {"json": {}}, 400),
        ("get", "/documents", {}, 400),
        ("get", "/embedding-models", {}, 200),
    ]

    for method, path, kwargs, allowed_status in protected_routes:
        request_fn = getattr(client, method)

        missing = request_fn(path, **kwargs)
        assert missing.status_code == 401
        assert missing.get_json()["error"] == "Missing API key."

        invalid = request_fn(path, headers={"x-api-key": "wrong"}, **kwargs)
        assert invalid.status_code == 403
        assert invalid.get_json()["error"] == "Invalid API key."

        allowed = request_fn(path, headers={"x-api-key": "secret"}, **kwargs)
        assert allowed.status_code == allowed_status


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
