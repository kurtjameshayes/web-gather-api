"""GET adaptive-feedback maps AuthorizationError to JSON 401/403.

These GET handlers catch AuthorizationError but not RuntimeError (uninitialized
config is a generic 500 in open PR #230). Route-level missing/invalid key and
role rejection were not pinned on base or that PR.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp


def _client():
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    return app.test_client()


def test_adaptive_feedback_missing_api_key_is_401() -> None:
    mock_config = load_config()
    mock_config.auth_required = True
    mock_config.api_key = "secret-key"
    mock_config.allowed_roles = []
    mock_critic = MagicMock()
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        response = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1"
        )
        log = client.get("/api/v4/compliance/adaptive-feedback/log")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."
    assert log.status_code == 401
    mock_critic.get_feedback_history.assert_not_called()
    mock_critic.get_feedback_log.assert_not_called()


def test_adaptive_feedback_invalid_api_key_is_403() -> None:
    mock_config = load_config()
    mock_config.auth_required = True
    mock_config.api_key = "secret-key"
    mock_config.allowed_roles = []
    mock_critic = MagicMock()
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        response = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1",
            headers={"x-api-key": "wrong"},
        )
    assert response.status_code == 403
    assert response.get_json()["error"] == "Invalid API key."
    mock_critic.get_feedback_history.assert_not_called()


def test_adaptive_feedback_insufficient_role_is_403() -> None:
    mock_config = load_config()
    mock_config.auth_required = True
    mock_config.api_key = "secret-key"
    mock_config.allowed_roles = ["admin"]
    mock_config.default_role_header = "x-role"
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        denied = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1",
            headers={"x-api-key": "secret-key", "x-role": "viewer"},
        )
        allowed = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1",
            headers={"x-api-key": "secret-key", "x-role": "Admin"},
        )
    assert denied.status_code == 403
    assert denied.get_json()["error"] == "Insufficient role."
    assert allowed.status_code == 200
    mock_critic.get_feedback_history.assert_called_once()
