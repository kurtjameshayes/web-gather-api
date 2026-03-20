"""Tests for compliance API v2, v3, v4 gap-analysis endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
import compliance_routes
from compliance_routes_v2 import compliance_v2_bp
from compliance_routes_v3 import compliance_v3_bp
from compliance_routes_v4 import compliance_v4_bp
from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisResponse, GapSummary


@pytest.fixture
def mock_config():
    """Config with auth disabled."""
    config = load_config()
    config.auth_required = False
    return config


@pytest.fixture
def mock_v2_service():
    """Mock for v2 gap analysis (suite.gap_analysis_v2)."""
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


@pytest.fixture
def mock_v3_service():
    """Mock for v3 gap analysis service."""
    mock = MagicMock()
    mock.run = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    return mock


@pytest.fixture
def mock_v4_service():
    """Mock for v4 gap analysis service."""
    mock = MagicMock()
    mock.run = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    return mock


@pytest.fixture
def mock_job_storage():
    """Mock job storage."""
    mock = MagicMock()
    mock.create_job = MagicMock(return_value="job-123")
    return mock


def test_v2_gap_analysis_success(mock_config, mock_v2_service) -> None:
    """POST /api/v2/compliance/gap-analysis returns result."""
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v2._get_suite_service", return_value=mock_v2_service
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v2/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"]},
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["policy_document_id"] == "pol-1"


def test_v2_gap_analysis_validation_error(mock_config, mock_v2_service) -> None:
    """POST /api/v2/compliance/gap-analysis without policy_document_id returns 422 or 500."""
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v2._get_suite_service", return_value=mock_v2_service
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
        client = app.test_client()
        response = client.post("/api/v2/compliance/gap-analysis", json={})
    assert response.status_code in (422, 500)


def test_v3_gap_analysis_sync_success(mock_config, mock_v3_service, mock_job_storage) -> None:
    """POST /api/v3/compliance/gap-analysis with run_async=False returns result."""
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_v3_service
    ), patch("compliance_routes_v3._get_job_storage", return_value=mock_job_storage):
        app = Flask(__name__)
        app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v3/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"], "run_async": False},
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["policy_document_id"] == "pol-1"


def test_v3_gap_analysis_async_returns_job_id(mock_config, mock_v3_service, mock_job_storage) -> None:
    """POST /api/v3/compliance/gap-analysis with run_async=True returns 202 and job_id."""
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock()
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_v3_service
    ), patch("compliance_routes_v3._get_job_storage", return_value=mock_job_storage), patch(
        "compliance_routes_v3._get_suite_service", return_value=mock_suite
    ), patch("compliance_routes_v3.start_gap_analysis_job", return_value="job-123"):
        app = Flask(__name__)
        app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v3/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"], "run_async": True},
        )
    assert response.status_code == 202
    data = response.get_json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_v4_gap_analysis_sync_success(mock_config, mock_v4_service, mock_job_storage) -> None:
    """POST /api/v4/compliance/gap-analysis with run_async=False returns result."""
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v4._get_gap_analysis_v4_service", return_value=mock_v4_service
    ), patch("compliance_routes_v4._get_job_storage", return_value=mock_job_storage):
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v4/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"], "run_async": False},
        )
    assert response.status_code == 200
    data = response.get_json()
    assert data["policy_document_id"] == "pol-1"


def test_v4_gap_analysis_async_returns_job_id(mock_config, mock_v4_service, mock_job_storage) -> None:
    """POST /api/v4/compliance/gap-analysis with run_async=True returns 202 and job_id."""
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock()
    with patch.object(compliance_routes, "_config", mock_config), patch(
        "compliance_routes_v4._get_gap_analysis_v4_service", return_value=mock_v4_service
    ), patch("compliance_routes_v4._get_job_storage", return_value=mock_job_storage), patch(
        "compliance_routes_v4._get_suite_service", return_value=mock_suite
    ), patch("compliance_routes_v4.start_gap_analysis_job", return_value="job-123"):
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v4/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"], "run_async": True},
        )
    assert response.status_code == 202
    data = response.get_json()
    assert "job_id" in data
    assert data["status"] == "pending"
