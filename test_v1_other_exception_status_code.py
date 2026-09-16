"""v1 POSTs besides gap-analysis/health-score map only typed service errors.

#236 pinned /gap-analysis and /health-score. The remaining v1 handlers use the
same pattern: ComplianceSuiteServiceError.status_code is honored, but a generic
Exception with status_code stays JSON 500 Internal server error.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_suite_service import ComplianceSuiteServiceError


class _StatusException(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# (path, json body, suite method name)
_V1_ENDPOINTS = [
    ("/api/compliance/applicability", {"policy_document_id": "pol-1"}, "applicability"),
    ("/api/compliance/citations", {"policy_document_id": "pol-1"}, "citations"),
    ("/api/compliance/drift-check", {}, "drift_check"),
    (
        "/api/compliance/report",
        {"policy_document_id": "pol-1", "format": "markdown", "source": "latest_stored"},
        "report",
    ),
    (
        "/api/compliance/suggest-policy",
        {
            "policy_text": "We collect data.",
            "gap_analysis_text": "Missing retention disclosure.",
            "gap_analysis_match": "retention",
            "statute_text": "Must disclose retention period.",
        },
        "suggest_policy",
    ),
    (
        "/api/compliance/multi-jurisdictional",
        {"applicable_jurisdictions": ["CA", "VA"]},
        "multi_jurisdictional",
    ),
    ("/api/compliance/risk-assessment", {"policy_document_id": "pol-1"}, "risk_assessment"),
]


def _post(path: str, body: dict, mock_suite: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ), patch.object(compliance_routes, "_get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        return app.test_client().post(path, json=body)


@pytest.mark.parametrize("path,body,method", _V1_ENDPOINTS)
def test_v1_typed_service_error_maps_status_code(path: str, body: dict, method: str) -> None:
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock()
    setattr(
        mock_suite,
        method,
        AsyncMock(
            side_effect=ComplianceSuiteServiceError(
                "Policy not found: pol-1", status_code=404
            )
        ),
    )
    response = _post(path, body, mock_suite)
    assert response.status_code == 404
    assert "Policy not found" in response.get_json()["error"]


@pytest.mark.parametrize("path,body,method", _V1_ENDPOINTS)
def test_v1_generic_exception_with_status_code_stays_500(path: str, body: dict, method: str) -> None:
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock()
    setattr(mock_suite, method, AsyncMock(side_effect=_StatusException("Rate limit exceeded", 429)))
    response = _post(path, body, mock_suite)
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal server error"
