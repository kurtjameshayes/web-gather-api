"""Regression tests for v1 gap-analysis and health-score async job starters.

Base tests always send run_async=False. The production default is run_async=True,
which starts a background job and must force save_results=False so the job
completion path is the only writer of compliance_results.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp


@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    return config


@pytest.fixture
def mock_suite_service():
    mock = MagicMock()
    mock.gap_analysis = MagicMock()
    mock.health_score = MagicMock()
    mock._storage = MagicMock()
    return mock


@pytest.fixture
def mock_job_storage():
    return MagicMock()


@pytest.fixture
def client(mock_config, mock_suite_service, mock_job_storage):
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite_service
    ), patch.object(compliance_routes, "_get_job_storage", return_value=mock_job_storage):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client()


def test_gap_analysis_omitted_run_async_starts_job(
    client, mock_suite_service
) -> None:
    """Omitting run_async uses the default True and returns 202 with a job_id."""
    with patch(
        "compliance_routes.start_gap_analysis_job", return_value="job-ga-default"
    ) as start_gap:
        response = client.post(
            "/api/compliance/gap-analysis",
            json={"policy_document_id": "pol-1"},
        )
    assert response.status_code == 202
    data = response.get_json()
    assert data["job_id"] == "job-ga-default"
    assert data["status"] == "pending"
    mock_suite_service.gap_analysis.assert_not_called()
    start_gap.assert_called_once()


def test_gap_analysis_async_forces_save_results_false(
    client, mock_suite_service, mock_job_storage
) -> None:
    """Async v1 gap-analysis jobs must not persist via the request save_results flag."""
    with patch(
        "compliance_routes.start_gap_analysis_job", return_value="job-ga"
    ) as start_gap:
        response = client.post(
            "/api/compliance/gap-analysis",
            json={
                "policy_document_id": "pol-1",
                "save_results": True,
                "run_async": True,
            },
        )
    assert response.status_code == 202
    kwargs = start_gap.call_args.kwargs
    assert kwargs["request_dict"]["save_results"] is False
    assert "run_async" not in kwargs["request_dict"]
    assert kwargs["request_dict"]["policy_document_id"] == "pol-1"
    assert kwargs["job_storage"] is mock_job_storage
    assert kwargs["run_gap_analysis_fn"] is mock_suite_service.gap_analysis
    assert kwargs["compliance_storage"] is mock_suite_service._storage
    mock_suite_service.gap_analysis.assert_not_called()


def test_health_score_omitted_run_async_starts_job(
    client, mock_suite_service
) -> None:
    """Omitting run_async on health-score uses the default True and returns 202."""
    with patch(
        "compliance_routes.start_health_score_job", return_value="job-hs-default"
    ) as start_hs:
        response = client.post(
            "/api/compliance/health-score",
            json={"policy_document_id": "pol-1"},
        )
    assert response.status_code == 202
    data = response.get_json()
    assert data["job_id"] == "job-hs-default"
    assert data["status"] == "pending"
    mock_suite_service.health_score.assert_not_called()
    start_hs.assert_called_once()


def test_health_score_async_forces_save_results_false(
    client, mock_suite_service, mock_job_storage
) -> None:
    """Async v1 health-score jobs must force save_results=False and use the health-score starter."""
    with patch(
        "compliance_routes.start_health_score_job", return_value="job-hs"
    ) as start_hs, patch(
        "compliance_routes.start_gap_analysis_job"
    ) as start_gap:
        response = client.post(
            "/api/compliance/health-score",
            json={
                "policy_document_id": "pol-1",
                "save_results": True,
                "run_async": True,
            },
        )
    assert response.status_code == 202
    start_gap.assert_not_called()
    kwargs = start_hs.call_args.kwargs
    assert kwargs["request_dict"]["save_results"] is False
    assert "run_async" not in kwargs["request_dict"]
    assert kwargs["request_dict"]["policy_document_id"] == "pol-1"
    assert kwargs["job_storage"] is mock_job_storage
    assert kwargs["run_health_score_fn"] is mock_suite_service.health_score
    assert kwargs["compliance_storage"] is mock_suite_service._storage
    mock_suite_service.health_score.assert_not_called()
