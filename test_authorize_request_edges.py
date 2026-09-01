"""Additional authorize_request contracts not covered by the happy-path role tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from compliance_config import load_config
from security import AuthorizationError, authorize_request


def test_authorize_request_skipped_when_auth_not_required() -> None:
    config = load_config()
    config.auth_required = False
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    request = SimpleNamespace(headers={})
    authorize_request(config, request)


def test_authorize_request_empty_configured_key_skips_compare() -> None:
    """When auth is required but config.api_key is empty, any non-empty header passes.

    The invalid-key check is `if config.api_key and not compare(...)`.
    """
    config = load_config()
    config.auth_required = True
    config.api_key = ""
    config.allowed_roles = []
    request = SimpleNamespace(headers={"x-api-key": "anything"})
    authorize_request(config, request)


def test_authorize_request_empty_configured_key_still_requires_header() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = ""
    request = SimpleNamespace(headers={})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, request)
    assert exc.value.status_code == 401
    assert exc.value.message == "Missing API key."


def test_authorize_request_strips_provided_key() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = []
    request = SimpleNamespace(headers={"x-api-key": "  secret  "})
    authorize_request(config, request)


def test_authorize_request_custom_role_header() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["admin"]
    config.default_role_header = "x-custom-role"
    missing = SimpleNamespace(headers={"x-api-key": "secret", "x-role": "admin"})
    with pytest.raises(AuthorizationError) as exc:
        authorize_request(config, missing)
    assert exc.value.status_code == 403
    assert exc.value.message == "Insufficient role."

    allowed = SimpleNamespace(headers={"x-api-key": "secret", "x-custom-role": "admin"})
    authorize_request(config, allowed)


def test_authorize_request_role_is_case_insensitive_and_stripped() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    config.allowed_roles = ["Admin"]
    request = SimpleNamespace(headers={"x-api-key": "secret", "x-role": "  ADMIN  "})
    authorize_request(config, request)
