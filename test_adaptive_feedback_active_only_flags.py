"""Regression tests for GET /adaptive-feedback active_only query coercion.

Base tests only pin `active_only=true`. The route lowercases the value and
accepts true/1/yes — not the broader `_to_bool` set (e.g. `on`).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp


@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    return config


@pytest.fixture
def client(mock_config):
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    return app.test_client()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", False),
        ("false", False),
        ("0", False),
    ],
)
def test_active_only_accepts_true_1_yes_but_not_on(
    client, mock_config, raw: str, expected: bool
) -> None:
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        response = client.get(
            f"/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&active_only={raw}"
        )
    assert response.status_code == 200
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="pol-1",
        active_only=expected,
        limit=50,
        offset=0,
    )


def test_gap_analysis_request_rejects_non_positive_num_rows() -> None:
    """num_rows is an optional positive cap on statute rows; 0 must not mean 'all'."""
    from pydantic import ValidationError

    from compliance_suite_schemas import GapAnalysisRequest

    GapAnalysisRequest(policy_document_id="policy-1", num_rows=1)
    with pytest.raises(ValidationError):
        GapAnalysisRequest(policy_document_id="policy-1", num_rows=0)
    with pytest.raises(ValidationError):
        GapAnalysisRequest(policy_document_id="policy-1", num_rows=-3)
