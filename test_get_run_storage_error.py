"""GET /runs/<id> storage-error mapping vs list-runs 502.

List endpoints catch storage exceptions as HTTP 502. Get-by-id does not, so a
Mongo failure is Flask's generic 500. A mapping drift here changes operator
alerting and client retry behavior for a single-run lookup.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

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
    storage.get_run_by_id = AsyncMock(return_value=None)
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


def test_get_run_storage_exception_is_generic_500_not_list_502(
    client, mock_storage
) -> None:
    """GET /runs/<id> does not catch storage errors (unlike GET /runs → 502)."""
    mock_storage.get_run_by_id.side_effect = RuntimeError("mongo unavailable")

    response = client.get("/api/compliance/runs/run-1")

    assert response.status_code == 500
    assert response.status_code != 502
    mock_storage.get_run_by_id.assert_awaited_once_with("run-1")
