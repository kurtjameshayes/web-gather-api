"""POST /api/v2/compliance/gap-analysis ignores run_async and always runs sync."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes_v2 import compliance_v2_bp
from compliance_suite_schemas import GapAnalysisResponse, GapSummary


def _client():
    app = Flask(__name__)
    app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
    return app.test_client()


def _mock_v2_service():
    mock = MagicMock()
    mock.gap_analysis_v2 = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    return mock


def test_v2_gap_analysis_run_async_true_still_runs_synchronously() -> None:
    """v2 has no job starter: run_async=True (the schema default) still returns 200."""
    mock_config = load_config()
    mock_config.auth_required = False
    mock_v2 = _mock_v2_service()
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v2._get_suite_service", return_value=mock_v2
    ), patch("compliance_job_service.start_gap_analysis_job") as start_job:
        client = _client()
        response = client.post(
            "/api/v2/compliance/gap-analysis",
            json={
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
                "run_async": True,
            },
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["policy_document_id"] == "pol-1"
    mock_v2.gap_analysis_v2.assert_awaited_once()
    start_job.assert_not_called()


def test_v2_gap_analysis_omitted_run_async_does_not_return_202() -> None:
    """Omitted run_async defaults True on the schema but v2 still executes inline."""
    mock_config = load_config()
    mock_config.auth_required = False
    mock_v2 = _mock_v2_service()
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v2._get_suite_service", return_value=mock_v2
    ), patch("compliance_job_service.start_gap_analysis_job") as start_job:
        client = _client()
        response = client.post(
            "/api/v2/compliance/gap-analysis",
            json={"policy_document_id": "pol-1"},
        )
    assert response.status_code == 200
    assert "job_id" not in response.get_json()
    start_job.assert_not_called()
    mock_v2.gap_analysis_v2.assert_awaited_once()
