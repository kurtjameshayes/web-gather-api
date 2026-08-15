"""Regression tests for GET /runs and GET /alerts query normalization.

Base endpoint tests only assert 200 + top-level keys. Storage listing is covered
in an open PR; these tests pin the route-layer clamps and ISO8601 validation
that decide what reaches storage.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
def mock_storage():
    storage = MagicMock()
    storage.list_runs = AsyncMock(return_value=([], 0))
    storage.list_alerts = AsyncMock(return_value=([], 0))
    return storage


@pytest.fixture
def app(mock_config, mock_storage):
    suite = MagicMock()
    suite._storage = mock_storage
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=suite
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_list_runs_clamps_limit_and_offset_and_parses_types(client, mock_storage) -> None:
    response = client.get(
        "/api/compliance/runs"
        "?policy_document_id=pol-1"
        "&since=2026-03-01T00:00:00Z"
        "&until=2026-03-31T23:59:59+00:00"
        "&limit=999"
        "&offset=-4"
        "&types=gap, health_score, ,multi_jurisdictional"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 200
    assert payload["offset"] == 0
    assert payload["total"] == 0
    assert payload["runs"] == []
    mock_storage.list_runs.assert_awaited_once_with(
        policy_document_id="pol-1",
        since="2026-03-01T00:00:00Z",
        until="2026-03-31T23:59:59+00:00",
        types=["gap", "health_score", "multi_jurisdictional"],
        limit=200,
        offset=0,
    )


def test_list_runs_invalid_limit_offset_default_and_blank_types(client, mock_storage) -> None:
    response = client.get("/api/compliance/runs?limit=abc&offset=nope&types=,,,")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 50
    assert payload["offset"] == 0
    mock_storage.list_runs.assert_awaited_once_with(
        policy_document_id=None,
        since=None,
        until=None,
        types=None,
        limit=50,
        offset=0,
    )


def test_list_runs_rejects_invalid_since_and_until(client, mock_storage) -> None:
    since_resp = client.get("/api/compliance/runs?since=not-a-date")
    assert since_resp.status_code == 400
    assert "since" in since_resp.get_json()["error"].lower()

    until_resp = client.get("/api/compliance/runs?until=2026-13-40")
    assert until_resp.status_code == 400
    assert "until" in until_resp.get_json()["error"].lower()
    mock_storage.list_runs.assert_not_called()


def test_list_runs_storage_failure_returns_502(client, mock_storage) -> None:
    mock_storage.list_runs.side_effect = RuntimeError("mongo unavailable")
    response = client.get("/api/compliance/runs")
    assert response.status_code == 502
    assert "mongo unavailable" in response.get_json()["error"]


def test_list_alerts_clamps_limit_and_forwards_filters(client, mock_storage) -> None:
    mock_storage.list_alerts.return_value = (
        [
            {
                "alert_id": "alert-1",
                "type": "regulatory_drift",
                "policy_document_id": "pol-1",
                "trigger": "statute_index_changed",
                "affected_jurisdictions": ["CA"],
                "detected_at": "2026-03-21T10:00:00+00:00",
            }
        ],
        1,
    )
    response = client.get(
        "/api/compliance/alerts"
        "?policy_document_id=pol-1"
        "&company_name=Acme"
        "&jurisdiction=CA"
        "&since=2026-03-01T00:00:00Z"
        "&limit=0"
        "&offset=3"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 0
    assert payload["offset"] == 3
    assert payload["total"] == 1
    assert payload["alerts"][0]["alert_id"] == "alert-1"
    mock_storage.list_alerts.assert_awaited_once_with(
        policy_document_id="pol-1",
        company_name="Acme",
        jurisdiction="CA",
        since="2026-03-01T00:00:00Z",
        limit=0,
        offset=3,
    )


def test_list_alerts_invalid_since_is_ignored_unlike_runs(client, mock_storage) -> None:
    """Alerts do not 400 on bad since; they drop the filter instead of rejecting."""
    response = client.get("/api/compliance/alerts?since=not-iso&limit=xx&offset=-1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["limit"] == 50
    assert payload["offset"] == 0
    mock_storage.list_alerts.assert_awaited_once_with(
        policy_document_id=None,
        company_name=None,
        jurisdiction=None,
        since=None,
        limit=50,
        offset=0,
    )


def test_list_alerts_storage_failure_returns_502(client, mock_storage) -> None:
    mock_storage.list_alerts.side_effect = RuntimeError("alerts down")
    response = client.get("/api/compliance/alerts")
    assert response.status_code == 502
    assert "alerts down" in response.get_json()["error"]
