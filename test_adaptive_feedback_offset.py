"""GET adaptive-feedback offset coercion (invalid defaults, negatives clamp to 0).

Limit clamps are covered by open PR #230. Offset on both history and log
handlers was still untested: invalid values default to 0, negatives become 0,
and a valid offset is forwarded to the critic.
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


def test_adaptive_feedback_invalid_offset_defaults_to_zero() -> None:
    mock_config = load_config()
    mock_config.auth_required = False
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        response = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&offset=abc"
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["offset"] == 0
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=50,
        offset=0,
    )


def test_adaptive_feedback_negative_offset_clamps_to_zero() -> None:
    mock_config = load_config()
    mock_config.auth_required = False
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    mock_critic.get_feedback_log.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        history = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&offset=-8"
        )
        log = client.get("/api/v4/compliance/adaptive-feedback/log?offset=-3")
    assert history.status_code == 200
    assert history.get_json()["offset"] == 0
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=50,
        offset=0,
    )
    assert log.status_code == 200
    assert log.get_json()["offset"] == 0
    mock_critic.get_feedback_log.assert_called_once_with(run_id=None, limit=50, offset=0)


def test_adaptive_feedback_positive_offset_is_forwarded() -> None:
    mock_config = load_config()
    mock_config.auth_required = False
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = [{"_id": "fb-3"}]
    mock_critic.get_feedback_log.return_value = [{"_id": "log-3"}]
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        history = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&offset=12"
        )
        log = client.get("/api/v4/compliance/adaptive-feedback/log?run_id=run-9&offset=4")
    assert history.status_code == 200
    assert history.get_json()["offset"] == 12
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=50,
        offset=12,
    )
    assert log.status_code == 200
    assert log.get_json()["offset"] == 4
    mock_critic.get_feedback_log.assert_called_once_with(run_id="run-9", limit=50, offset=4)
