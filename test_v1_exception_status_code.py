"""v1 compliance POSTs map ComplianceSuiteServiceError.status_code only.

A generic Exception that happens to have status_code is still JSON 500
"Internal server error". Distinct from v2/v3/v4, which inspect hasattr(exc,
"status_code") on the bare Exception handler.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_suite_schemas import GapAnalysisResponse, GapSummary
from compliance_suite_service import ComplianceSuiteServiceError


class _StatusException(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _post(path: str, json_body: dict, mock_suite: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ), patch.object(compliance_routes, "_get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        return app.test_client().post(path, json=json_body)


def test_v1_gap_analysis_maps_typed_service_error() -> None:
    mock_suite = MagicMock()
    mock_suite.gap_analysis = AsyncMock(
        side_effect=ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)
    )
    mock_suite._storage = MagicMock()
    response = _post(
        "/api/compliance/gap-analysis",
        {"policy_document_id": "pol-1", "run_async": False},
        mock_suite,
    )
    assert response.status_code == 429
    assert response.get_json()["error"] == "Rate limit exceeded"


def test_v1_gap_analysis_generic_status_code_stays_500() -> None:
    """hasattr(exc, "status_code") is ignored for untyped exceptions on v1."""
    mock_suite = MagicMock()
    mock_suite.gap_analysis = AsyncMock(
        side_effect=_StatusException("Rate limit exceeded", 429)
    )
    mock_suite._storage = MagicMock()
    response = _post(
        "/api/compliance/gap-analysis",
        {"policy_document_id": "pol-1", "run_async": False},
        mock_suite,
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal server error"


def test_v1_health_score_typed_error_maps_and_plain_stays_500() -> None:
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock()
    mock_suite.health_score = AsyncMock(
        side_effect=ComplianceSuiteServiceError("Policy not found: pol-x", status_code=404)
    )
    typed = _post(
        "/api/compliance/health-score",
        {"policy_document_id": "pol-x", "run_async": False},
        mock_suite,
    )
    assert typed.status_code == 404
    assert "Policy not found" in typed.get_json()["error"]

    mock_suite.health_score = AsyncMock(side_effect=_StatusException("boom", 404))
    plain = _post(
        "/api/compliance/health-score",
        {"policy_document_id": "pol-x", "run_async": False},
        mock_suite,
    )
    assert plain.status_code == 500
    assert plain.get_json()["error"] == "Internal server error"


def test_v1_gap_analysis_success_still_returns_result() -> None:
    """Sanity: the typed/generic split does not break the success path used by base tests."""
    mock_suite = MagicMock()
    mock_suite.gap_analysis = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    mock_suite._storage = MagicMock()
    response = _post(
        "/api/compliance/gap-analysis",
        {"policy_document_id": "pol-1", "run_async": False},
        mock_suite,
    )
    assert response.status_code == 200
    assert response.get_json()["policy_document_id"] == "pol-1"
