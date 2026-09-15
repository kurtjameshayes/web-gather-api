"""v2 gap-analysis maps generic Exception.status_code; plain exceptions stay JSON 500.

Distinct from open PR #235, which pins the same mapping on v3/v4 only. v1 does
not map hasattr(exc, "status_code") on generic Exception (see
test_v1_exception_status_code.py).
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes_v2 import compliance_v2_bp


class _StatusException(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _post_v2(mock_service: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch(
        "compliance_routes_v2._get_suite_service", return_value=mock_service
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
        client = app.test_client()
        return client.post(
            "/api/v2/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"]},
        )


def test_v2_gap_analysis_maps_generic_exception_status_code() -> None:
    """A generic Exception with status_code=429 is returned as HTTP 429, not Flask 500."""
    mock_service = MagicMock()
    mock_service.gap_analysis_v2 = AsyncMock(
        side_effect=_StatusException("Rate limit exceeded", 429)
    )
    response = _post_v2(mock_service)
    assert response.status_code == 429
    assert response.get_json()["error"] == "Rate limit exceeded"
    mock_service.gap_analysis_v2.assert_awaited_once()


def test_v2_gap_analysis_maps_404_status_code() -> None:
    mock_service = MagicMock()
    mock_service.gap_analysis_v2 = AsyncMock(
        side_effect=_StatusException("Policy not found: pol-missing", 404)
    )
    response = _post_v2(mock_service)
    assert response.status_code == 404
    assert "Policy not found" in response.get_json()["error"]


def test_v2_gap_analysis_plain_exception_is_json_500() -> None:
    """Exceptions without status_code remain JSON 500 Internal server error."""
    mock_service = MagicMock()
    mock_service.gap_analysis_v2 = AsyncMock(side_effect=RuntimeError("mongo down"))
    response = _post_v2(mock_service)
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal server error"
