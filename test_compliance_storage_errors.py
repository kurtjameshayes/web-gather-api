"""Regression tests for compliance list/get storage error mapping.

GET /runs and GET /alerts map storage exceptions to HTTP 502.
GET /runs/<id> returns the raw storage document on success.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

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
    mock._storage = MagicMock()
    mock._storage.list_runs = AsyncMock(return_value=([], 0))
    mock._storage.get_run_by_id = AsyncMock(return_value=None)
    mock._storage.list_alerts = AsyncMock(return_value=([], 0))
    return mock


@pytest.fixture
def client(mock_config, mock_suite_service):
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite_service
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client()


def test_list_runs_storage_exception_is_502(client, mock_suite_service) -> None:
    mock_suite_service._storage.list_runs = AsyncMock(side_effect=RuntimeError("mongo down"))
    response = client.get("/api/compliance/runs")
    assert response.status_code == 502
    assert "mongo down" in response.get_json()["error"]


def test_list_alerts_storage_exception_is_502(client, mock_suite_service) -> None:
    mock_suite_service._storage.list_alerts = AsyncMock(side_effect=RuntimeError("alerts unavailable"))
    response = client.get("/api/compliance/alerts")
    assert response.status_code == 502
    assert "alerts unavailable" in response.get_json()["error"]


def test_get_run_success_returns_raw_document(client, mock_suite_service) -> None:
    """Successful GET /runs/<id> returns the storage document as-is (not a schema dump)."""
    raw = {
        "run_id": "run-9",
        "type": "gap_analysis",
        "extra_field": "preserved",
        "nested": {"ok": True},
    }
    mock_suite_service._storage.get_run_by_id = AsyncMock(return_value=raw)
    response = client.get("/api/compliance/runs/run-9")
    assert response.status_code == 200
    assert response.get_json() == raw
    mock_suite_service._storage.get_run_by_id.assert_awaited_once_with("run-9")
