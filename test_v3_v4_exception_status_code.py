"""v3/v4 gap-analysis maps generic exceptions that carry status_code.

v1 ComplianceSuiteServiceError and typed V3/V4 service errors are already
covered. These handlers also inspect hasattr(exc, 'status_code') on any
Exception so a 404/429 from a wrapped error is not collapsed to HTTP 500.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes_v3 import compliance_v3_bp
from compliance_routes_v4 import compliance_v4_bp


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    return config


def test_v3_generic_exception_with_status_code_is_not_500(mock_config) -> None:
    mock_v3 = MagicMock()
    mock_v3.run = AsyncMock(side_effect=_StatusError("Policy not found: pol-1", 404))
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_v3
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v3/compliance/gap-analysis",
            json={
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
                "run_async": False,
            },
        )
    assert response.status_code == 404
    assert "Policy not found" in response.get_json()["error"]


def test_v4_generic_exception_with_status_code_is_not_500(mock_config) -> None:
    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock(side_effect=_StatusError("Rate limit exceeded", 429))
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v4._get_gap_analysis_v4_service", return_value=mock_v4
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v4/compliance/gap-analysis",
            json={
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
                "run_async": False,
            },
        )
    assert response.status_code == 429
    assert "Rate limit" in response.get_json()["error"]


def test_v3_plain_exception_is_internal_500(mock_config) -> None:
    mock_v3 = MagicMock()
    mock_v3.run = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_v3
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v3/compliance/gap-analysis",
            json={
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
                "run_async": False,
            },
        )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal server error"
