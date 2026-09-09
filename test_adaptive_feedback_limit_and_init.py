"""GET adaptive-feedback pagination clamps and uninitialized-config error shape."""
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


def test_adaptive_feedback_limit_zero_clamps_to_one() -> None:
    """Unlike GET /runs and /alerts (limit=0 stays 0), adaptive-feedback clamps to at least 1."""
    mock_config = load_config()
    mock_config.auth_required = False
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        response = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&limit=0"
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["limit"] == 1
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=1,
        offset=0,
    )


def test_adaptive_feedback_limit_above_200_clamps_and_invalid_defaults() -> None:
    """limit=500 becomes 200; non-integer limit falls back to 50."""
    mock_config = load_config()
    mock_config.auth_required = False
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    mock_critic.get_feedback_log.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        client = _client()
        high = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&limit=500"
        )
        invalid = client.get(
            "/api/v4/compliance/adaptive-feedback/log?limit=abc"
        )
    assert high.status_code == 200
    assert high.get_json()["limit"] == 200
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=False,
        limit=200,
        offset=0,
    )
    assert invalid.status_code == 200
    assert invalid.get_json()["limit"] == 50
    mock_critic.get_feedback_log.assert_called_once_with(run_id=None, limit=50, offset=0)


def test_adaptive_feedback_uninitialized_config_is_generic_500() -> None:
    """GET adaptive-feedback does not catch RuntimeError; Flask returns a generic 500.

    Distinct from POST v2/v3/v4 gap-analysis, which map uninitialized config to JSON 500.
    """
    previous = compliance_routes._config
    previous_critic = compliance_routes._critic_service
    compliance_routes._config = None
    compliance_routes._critic_service = MagicMock()
    try:
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        client = app.test_client()
        response = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1"
        )
        log_response = client.get("/api/v4/compliance/adaptive-feedback/log")
    finally:
        compliance_routes._config = previous
        compliance_routes._critic_service = previous_critic

    assert response.status_code == 500
    payload = response.get_json(silent=True)
    assert payload is None or payload.get("error") != (
        "Compliance not initialized; call init_compliance first"
    )
    assert log_response.status_code == 500
    log_payload = log_response.get_json(silent=True)
    assert log_payload is None or log_payload.get("error") != (
        "Compliance not initialized; call init_compliance first"
    )
