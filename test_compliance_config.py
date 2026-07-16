"""Regression tests for environment-driven compliance configuration."""
from __future__ import annotations

from typing import Any

import pytest

from compliance_config import load_config


@pytest.fixture(autouse=True)
def isolated_auth_environment(monkeypatch: Any, tmp_path: Any) -> None:
    """Keep host credentials and repository config out of auth tests."""
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("COMPLIANCE_API_KEY", raising=False)
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.mark.parametrize("key_variable", ["COMPLIANCE_API_KEY", "API_KEY"])
def test_load_config_auto_enables_auth_when_api_key_is_set(
    monkeypatch: Any,
    key_variable: str,
) -> None:
    monkeypatch.setenv(key_variable, "configured-secret")

    config = load_config()

    assert config.api_key == "configured-secret"
    assert config.auth_required is True


def test_load_config_explicit_auth_setting_overrides_auto_enable(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("COMPLIANCE_API_KEY", "configured-secret")
    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "false")

    config = load_config()

    assert config.api_key == "configured-secret"
    assert config.auth_required is False


def test_load_config_without_api_key_keeps_auth_disabled() -> None:
    config = load_config()

    assert config.api_key == ""
    assert config.auth_required is False
