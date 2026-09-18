"""Pin GET /alerts jurisdiction empty vs whitespace filter contracts.

Empty `jurisdiction=` is falsy after `or None` and is unfiltered. Whitespace is
forwarded to storage as an equality on affected_jurisdictions (storage does not
strip jurisdiction the way it ignores whitespace-only company_name).
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
def mock_suite_service():
    mock = MagicMock()
    mock._storage = MagicMock()
    mock._storage.list_alerts = AsyncMock(return_value=([], 0))
    return mock


@pytest.fixture
def app(mock_config, mock_suite_service):
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_suite_service", return_value=mock_suite_service
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_alerts_empty_jurisdiction_is_unfiltered(client, mock_suite_service) -> None:
    response = client.get("/api/compliance/alerts?jurisdiction=")
    assert response.status_code == 200
    kwargs = mock_suite_service._storage.list_alerts.await_args.kwargs
    assert kwargs["jurisdiction"] is None


def test_alerts_whitespace_jurisdiction_is_forwarded(client, mock_suite_service) -> None:
    response = client.get("/api/compliance/alerts?jurisdiction=%20%20%20")
    assert response.status_code == 200
    kwargs = mock_suite_service._storage.list_alerts.await_args.kwargs
    assert kwargs["jurisdiction"] == "   "
