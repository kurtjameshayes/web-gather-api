"""GET /index-jobs/<id> does not catch storage exceptions.

List endpoints such as GET /runs and GET /alerts map Mongo failures to JSON
502. GET /api/compliance/jobs/<id> is a generic HTTP 500 (#239). Index-job
GET is the same uncaught path: Flask's test client turns the exception into
a generic HTTP 500, not JSON 502.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    init_core(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_get_index_job_storage_error_is_generic_500(client) -> None:
    """Storage raise is Flask's generic 500, not the JSON 502 used by list routes."""
    mock_storage = MagicMock()
    mock_storage.get_job.side_effect = RuntimeError("mongo unavailable")

    with patch("core._get_index_job_storage", return_value=mock_storage):
        response = client.get("/index-jobs/job-123")

    assert response.status_code == 500
    body = response.get_json()
    # Generic Flask 500 has no JSON 502 envelope.
    if body is not None:
        assert body.get("error") != "mongo unavailable"
        assert response.status_code != 502
