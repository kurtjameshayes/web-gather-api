"""COMPLIANCE_ALLOWED_ROLES blank-item split and the empty-list role-skip contract."""
from __future__ import annotations

from types import SimpleNamespace

from compliance_config import load_config
from security import authorize_request


def test_allowed_roles_skips_blank_csv_items(monkeypatch) -> None:
    monkeypatch.setenv("COMPLIANCE_ALLOWED_ROLES", "admin, , compliance, ")
    monkeypatch.delenv("COMPLIANCE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)

    config = load_config()
    assert config.allowed_roles == ["admin", "compliance"]


def test_allowed_roles_only_blanks_disables_role_check(monkeypatch) -> None:
    """A truthy env value of only commas/spaces stores [] and skips role enforcement."""
    monkeypatch.setenv("COMPLIANCE_ALLOWED_ROLES", ", , ")
    monkeypatch.delenv("COMPLIANCE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("COMPLIANCE_AUTH_REQUIRED", raising=False)

    config = load_config()
    assert config.allowed_roles == []

    config.auth_required = True
    config.api_key = "secret"
    authorize_request(
        config,
        SimpleNamespace(headers={"x-api-key": "secret", "x-role": "viewer"}),
    )
