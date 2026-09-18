"""Pin GET /jobs/<id> storage exception mapping.

GET /runs and GET /alerts map storage failures to JSON 502. GET /jobs/<id> calls
get_job after auth with no try/except, so Flask's test client turns an unhandled
exception into a generic HTTP 500 (not JSON 502).
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
def mock_job_storage():
    mock = MagicMock()
    mock.get_job.side_effect = RuntimeError("mongo unavailable")
    return mock


@pytest.fixture
def app(mock_config, mock_job_storage):
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_get_job_storage", return_value=mock_job_storage
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_get_job_storage_error_is_generic_http_500(client, mock_job_storage) -> None:
    response = client.get("/api/compliance/jobs/job-1")
    assert response.status_code == 500
    mock_job_storage.get_job.assert_called_once_with("job-1")
    payload = response.get_json(silent=True)
    # List endpoints map this to JSON 502 {"error": "..."}. get-by-id does not.
    if payload is not None:
        assert payload.get("error") != "mongo unavailable"