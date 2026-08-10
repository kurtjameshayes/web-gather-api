"""Regression tests for compliance config file loading and merge behavior."""
from __future__ import annotations

import json
from typing import Any

import pytest

from compliance_config import _load_from_file, load_config


def test_load_from_file_missing_path_returns_empty_dict(tmp_path: Any) -> None:
    """Missing config files must not crash startup; defaults remain available."""
    missing = tmp_path / "does-not-exist.json"
    assert _load_from_file(str(missing)) == {}


def test_load_from_file_invalid_json_returns_empty_dict(tmp_path: Any) -> None:
    """Corrupt JSON must fail closed to an empty overlay instead of raising."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not-valid-json", encoding="utf-8")
    assert _load_from_file(str(bad)) == {}


def test_load_from_file_reads_valid_json_object(tmp_path: Any) -> None:
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"top_k_statutes": 11, "compliance_database": "pc-test"}), encoding="utf-8")
    assert _load_from_file(str(path)) == {
        "top_k_statutes": 11,
        "compliance_database": "pc-test",
    }


@pytest.fixture
def isolated_config_env(monkeypatch: Any, tmp_path: Any):
    """Avoid host env / repo JSON leaking into file-merge assertions."""
    monkeypatch.delenv("COMPLIANCE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("COMPLIANCE_TOP_K_STATUTES", raising=False)
    monkeypatch.delenv("COMPLIANCE_ALLOWED_ROLES", raising=False)
    monkeypatch.delenv("ADAPTIVE_FEEDBACK_ENABLED", raising=False)
    return tmp_path


def test_load_config_merges_valid_file_overrides(
    monkeypatch: Any,
    isolated_config_env: Any,
) -> None:
    """File values should override defaults when present and well-formed."""
    config_path = isolated_config_env / "policy.json"
    config_path.write_text(
        json.dumps(
            {
                "top_k_statutes": 17,
                "compliance_database": "privacy-compliance-custom",
                "adaptive_feedback_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(config_path))

    config = load_config()

    assert config.top_k_statutes == 17
    assert config.compliance_database == "privacy-compliance-custom"
    assert config.adaptive_feedback_enabled is False


def test_load_config_uses_defaults_when_file_invalid(
    monkeypatch: Any,
    isolated_config_env: Any,
) -> None:
    """Invalid config files should leave DEFAULT_CONFIG values intact."""
    config_path = isolated_config_env / "broken.json"
    config_path.write_text('{"top_k_statutes": ', encoding="utf-8")
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(config_path))

    config = load_config()

    # Defaults from compliance_config.DEFAULT_CONFIG
    assert config.top_k_statutes == 5
    assert config.compliance_database == "privacy-compliance"
    assert config.adaptive_feedback_enabled is True
