"""GET /runs and GET /alerts empty vs whitespace policy_document_id filters.

`request.args.get(...) or None` treats "" as unfiltered but passes through
whitespace, which would otherwise match no Mongo documents.
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
def client(mock_config, mock_storage):
    suite = MagicMock()
    suite._storage = mock_storage
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=suite
    ), patch.object(compliance_routes, "_get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client(), mock_storage


def test_runs_empty_policy_document_id_is_unfiltered(client) -> None:
    http, storage = client
    response = http.get("/api/compliance/runs?policy_document_id=")
    assert response.status_code == 200
    kwargs = storage.list_runs.await_args.kwargs
    assert kwargs["policy_document_id"] is None


def test_runs_whitespace_policy_document_id_is_passed_through(client) -> None:
    http, storage = client
    response = http.get("/api/compliance/runs?policy_document_id=%20%20")
    assert response.status_code == 200
    kwargs = storage.list_runs.await_args.kwargs
    assert kwargs["policy_document_id"] == "  "


def test_alerts_empty_policy_document_id_is_unfiltered(client) -> None:
    http, storage = client
    response = http.get("/api/compliance/alerts?policy_document_id=")
    assert response.status_code == 200
    kwargs = storage.list_alerts.await_args.kwargs
    assert kwargs["policy_document_id"] is None


def test_alerts_whitespace_company_name_reaches_storage(client) -> None:
    """Route-layer `or None` keeps whitespace; storage later ignores strip-empty names."""
    http, storage = client
    response = http.get("/api/compliance/alerts?company_name=%20%20")
    assert response.status_code == 200
    kwargs = storage.list_alerts.await_args.kwargs
    assert kwargs["company_name"] == "  "
