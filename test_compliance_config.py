"""Tests for compliance configuration loading and environment overrides."""
from __future__ import annotations

from pathlib import Path

from compliance_config import load_config


def _use_empty_config_file(monkeypatch, tmp_path: Path) -> None:
    """Force load_config to ignore any local policy_compliance_config.json."""
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(tmp_path / "missing.json"))


def test_load_config_auto_enables_auth_when_api_key_is_configured(monkeypatch, tmp_path) -> None:
    _use_empty_config_file(monkeypatch, tmp_path)
    monkeypatch.setenv("COMPLIANCE_API_KEY", "secret-key")
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)

    config = load_config()

    assert config.api_key == "secret-key"
    assert config.auth_required is True


def test_load_config_respects_explicit_auth_required_override(monkeypatch, tmp_path) -> None:
    _use_empty_config_file(monkeypatch, tmp_path)
    monkeypatch.setenv("COMPLIANCE_API_KEY", "secret-key")
    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "false")

    config = load_config()

    assert config.api_key == "secret-key"
    assert config.auth_required is False


def test_load_config_applies_adaptive_feedback_environment_overrides(monkeypatch, tmp_path) -> None:
    _use_empty_config_file(monkeypatch, tmp_path)
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "off")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "17")

    config = load_config()

    assert config.adaptive_feedback_enabled is False
    assert config.adaptive_feedback_max_items == 17
