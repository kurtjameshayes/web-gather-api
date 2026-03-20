"""Tests for compliance API v1 endpoints (applicability, gap-analysis, jobs, runs, etc.)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_routes import compliance_bp
from compliance_config import load_config
from compliance_suite_schemas import (
    ApplicabilityResponse,
    GapAnalysisResponse,
    GapSummary,
    MultiJurisdictionalResponse,
    HealthScoreResponse,
    DriftCheckResponse,
)


@pytest.fixture
def mock_config():
    """Config with auth disabled for testing."""
    config = load_config()
    config.auth_required = False
    return config


@pytest.fixture
def mock_suite_service():
    """Mock ComplianceSuiteService with async methods returning Pydantic models."""
    mock = MagicMock()
    mock.applicability = AsyncMock(
        return_value=ApplicabilityResponse(applicable_jurisdictions=["CA"], confidence={"CA": 0.9})
    )
    mock.gap_analysis = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    mock.multi_jurisdictional = AsyncMock(
        return_value=MultiJurisdictionalResponse(
            applicable_jurisdictions=["CA", "VA"],
            analyzed_at="2026-03-18T00:00:00Z",
            strictest_common_denominator=[],
            conflicts_between_jurisdictions=[],
        )
    )
    mock.health_score = AsyncMock(
        return_value=HealthScoreResponse(
            policy_document_id="pol-1",
            privacy_health_score=85,
            analyzed_at="2026-03-18T00:00:00Z",
        )
    )
    mock.drift_check = AsyncMock(
        return_value=DriftCheckResponse(alerts=[], policies_checked=1, alerts_written=0)
    )
    mock.report = AsyncMock(return_value={"sections": [], "policy_document_id": "pol-1"})
    mock._storage = MagicMock()
    mock._storage.list_runs = AsyncMock(return_value=([], 0))
    mock._storage.get_run_by_id = AsyncMock(return_value=None)
    mock._storage.list_alerts = AsyncMock(return_value=([], 0))
    mock.citations = AsyncMock(return_value=MagicMock(model_dump=lambda: {"citations": []}))
    mock.risk_assessment = AsyncMock(return_value=MagicMock(model_dump=lambda: {"assessment_id": "a1"}))
    mock.list_templates = AsyncMock(return_value=MagicMock(model_dump=lambda: {"templates": []}))
    mock.suggest_policy = AsyncMock(return_value=MagicMock(model_dump=lambda: {"suggestions": []}))
    mock.consumer_rights_router = AsyncMock(
        return_value=MagicMock(model_dump=lambda: {"route": "general", "suggestions": []})
    )
    return mock


@pytest.fixture
def mock_job_storage():
    """Mock ComplianceJobStorage."""
    mock = MagicMock()
    mock.get_job.return_value = None
    return mock


@pytest.fixture
def app(mock_config, mock_suite_service, mock_job_storage):
    """Flask app with compliance blueprint."""
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite_service
    ), patch.object(compliance_routes, "_get_job_storage", return_value=mock_job_storage):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app


@pytest.fixture
def client(app):
    """Test client."""
    return app.test_client()


def test_applicability_success(client, mock_suite_service) -> None:
    """POST /api/compliance/applicability returns jurisdictions."""
    response = client.post(
        "/api/compliance/applicability",
        json={"policy_document_id": "pol-1"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "applicable_jurisdictions" in data
    assert "CA" in data["applicable_jurisdictions"]


def test_applicability_validation_error(client) -> None:
    """POST /api/compliance/applicability without policy_id or text returns 422 or 500."""
    response = client.post("/api/compliance/applicability", json={})
    assert response.status_code in (422, 500)


def test_gap_analysis_success(client, mock_suite_service) -> None:
    """POST /api/compliance/gap-analysis returns gap analysis."""
    mock_suite_service.gap_analysis.return_value = GapAnalysisResponse(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-18T00:00:00Z",
        gaps=[],
        summary=GapSummary(total_requirements=1, missing=0, addressed=1, conflicts=0, partial=0, ambiguous=0, analysis_failures=0),
    )
    response = client.post(
        "/api/compliance/gap-analysis",
        json={"policy_document_id": "pol-1", "run_async": False},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["policy_document_id"] == "pol-1"
    assert "gaps" in data


def test_gap_analysis_validation_error(client) -> None:
    """POST /api/compliance/gap-analysis without policy_document_id returns 422 or 500."""
    response = client.post("/api/compliance/gap-analysis", json={"run_async": False})
    assert response.status_code in (422, 500)


def test_jobs_not_found(client, mock_job_storage) -> None:
    """GET /api/compliance/jobs/<id> returns 404 when job not found."""
    mock_job_storage.get_job.return_value = None
    response = client.get("/api/compliance/jobs/nonexistent")
    assert response.status_code == 404


def test_jobs_found(client, mock_job_storage) -> None:
    """GET /api/compliance/jobs/<id> returns job when found."""
    mock_job_storage.get_job.return_value = {"job_id": "j1", "status": "completed"}
    response = client.get("/api/compliance/jobs/j1")
    assert response.status_code == 200
    assert response.get_json()["job_id"] == "j1"


def test_multi_jurisdictional_success(client, mock_suite_service) -> None:
    """POST /api/compliance/multi-jurisdictional returns result."""
    response = client.post(
        "/api/compliance/multi-jurisdictional",
        json={"applicable_jurisdictions": ["CA", "VA"]},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "applicable_jurisdictions" in data or "strictest_common_denominator" in data


def test_health_score_success(client, mock_suite_service) -> None:
    """POST /api/compliance/health-score returns score."""
    response = client.post(
        "/api/compliance/health-score",
        json={"policy_document_id": "pol-1", "run_async": False},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "privacy_health_score" in data or "policy_document_id" in data


def test_drift_check_success(client, mock_suite_service) -> None:
    """POST /api/compliance/drift-check returns alerts."""
    response = client.post(
        "/api/compliance/drift-check",
        json={"policy_document_id": "pol-1"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "alerts" in data


def test_report_success(client, mock_suite_service) -> None:
    """POST /api/compliance/report returns report."""
    response = client.post(
        "/api/compliance/report",
        json={"policy_document_id": "pol-1", "format": "markdown", "source": "latest_stored"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "sections" in data or "policy_document_id" in data


def test_runs_list_success(client, mock_suite_service) -> None:
    """GET /api/compliance/runs returns runs list."""
    response = client.get("/api/compliance/runs")
    assert response.status_code == 200
    data = response.get_json()
    assert "runs" in data or "total" in data


def test_runs_get_not_found(client, mock_suite_service) -> None:
    """GET /api/compliance/runs/<id> returns 404 when run not found."""
    mock_suite_service._storage.get_run_by_id = AsyncMock(return_value=None)
    response = client.get("/api/compliance/runs/run-1")
    assert response.status_code == 404


def test_citations_success(client, mock_suite_service) -> None:
    """POST /api/compliance/citations returns citations."""
    response = client.post(
        "/api/compliance/citations",
        json={"policy_document_id": "pol-1"},
    )
    assert response.status_code == 200


def test_risk_assessment_success(client, mock_suite_service) -> None:
    """POST /api/compliance/risk-assessment returns assessment."""
    response = client.post(
        "/api/compliance/risk-assessment",
        json={"policy_document_id": "pol-1", "template_id": "t1"},
    )
    assert response.status_code == 200


def test_risk_assessment_templates_success(client, mock_suite_service) -> None:
    """GET /api/compliance/risk-assessment/templates returns templates."""
    response = client.get("/api/compliance/risk-assessment/templates")
    assert response.status_code == 200


def test_alerts_success(client, mock_suite_service) -> None:
    """GET /api/compliance/alerts returns alerts."""
    response = client.get("/api/compliance/alerts")
    assert response.status_code == 200


def test_suggest_policy_success(client, mock_suite_service) -> None:
    """POST /api/compliance/suggest-policy returns suggestions."""
    response = client.post(
        "/api/compliance/suggest-policy",
        json={
            "policy_text": "We collect data.",
            "gap_analysis_text": "Missing retention disclosure.",
            "gap_analysis_match": "retention",
            "statute_text": "Must disclose retention period.",
        },
    )
    assert response.status_code == 200


def test_consumer_rights_router_success(client, mock_suite_service) -> None:
    """POST /api/compliance/consumer-rights-router returns route."""
    response = client.post(
        "/api/compliance/consumer-rights-router",
        json={"text": "I want to delete my data"},
    )
    assert response.status_code == 200
