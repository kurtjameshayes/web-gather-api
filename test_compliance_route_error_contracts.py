"""Regression tests for leftover compliance route error/response contracts.

GET /runs/<id> returns the raw storage document (not wrapped). v2/v3/v4
authorize_request RuntimeError (uninitialized config) is JSON 500 with the
exception message, not Flask's generic HTML 500.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_routes_v2 import compliance_v2_bp
from compliance_routes_v3 import compliance_v3_bp
from compliance_routes_v4 import compliance_v4_bp
from compliance_suite_schemas import GapAnalysisResponse, GapSummary


@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    return config


def test_get_run_returns_raw_storage_document(mock_config) -> None:
    """GET /api/compliance/runs/<id> jsonifies the storage doc as-is (not wrapped)."""
    raw_doc = {
        "run_id": "run-1",
        "_id": "mongo-oid",
        "policy_document_id": "pol-1",
        "gaps": [{"status": "missing"}],
    }
    mock_suite = MagicMock()
    mock_suite._storage.get_run_by_id = AsyncMock(return_value=raw_doc)
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        client = app.test_client()
        response = client.get("/api/compliance/runs/run-1")
    assert response.status_code == 200
    assert response.get_json() == raw_doc
    mock_suite._storage.get_run_by_id.assert_awaited_once_with("run-1")


def test_v2_uninitialized_config_is_json_500() -> None:
    """v2 authorize_request RuntimeError maps to JSON 500 with the exception text."""
    mock_v2 = MagicMock()
    mock_v2.gap_analysis_v2 = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-03-18T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
        )
    )
    with patch(
        "compliance_routes_v2._get_config",
        side_effect=RuntimeError("Compliance config not initialized."),
    ), patch("compliance_routes_v2._get_suite_service", return_value=mock_v2):
        app = Flask(__name__)
        app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
        client = app.test_client()
        response = client.post(
            "/api/v2/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"]},
        )
    assert response.status_code == 500
    assert response.get_json() == {"error": "Compliance config not initialized."}
    mock_v2.gap_analysis_v2.assert_not_called()


def test_v3_uninitialized_config_is_json_500() -> None:
    """v3 authorize_request RuntimeError maps to JSON 500 before the service runs."""
    mock_v3 = MagicMock()
    mock_v3.run = AsyncMock()
    with patch(
        "compliance_routes_v3._get_config",
        side_effect=RuntimeError("Compliance config not initialized."),
    ), patch(
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
    assert response.get_json() == {"error": "Compliance config not initialized."}
    mock_v3.run.assert_not_called()


def test_v4_uninitialized_config_is_json_500() -> None:
    """v4 authorize_request RuntimeError maps to JSON 500 before the service runs."""
    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock()
    with patch(
        "compliance_routes_v4._get_config",
        side_effect=RuntimeError("Compliance config not initialized."),
    ), patch(
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
    assert response.status_code == 500
    assert response.get_json() == {"error": "Compliance config not initialized."}
    mock_v4.run.assert_not_called()
