"""GET /runs empty/whitespace since|until is 400; GET /alerts drops the filter.

#209 pins invalid calendar strings. Empty and whitespace still parse to None
via _parse_iso8601, but only /runs then 400s because since_arg is not None.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.list_runs = AsyncMock(return_value=([], 0))
    storage.list_alerts = AsyncMock(return_value=([], 0))
    return storage


@pytest.fixture
def client(mock_storage):
    config = load_config()
    config.auth_required = False
    suite = MagicMock()
    suite._storage = mock_storage
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=suite
    ), patch.object(compliance_routes, "_get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client()


@pytest.mark.parametrize("query", ["since=", "since=%20%20%20", "until=", "until=%20%20"])
def test_list_runs_empty_or_whitespace_since_until_is_400(client, mock_storage, query: str) -> None:
    response = client.get(f"/api/compliance/runs?{query}")
    assert response.status_code == 400
    err = response.get_json()["error"].lower()
    assert "iso8601" in err
    mock_storage.list_runs.assert_not_called()


def test_list_alerts_empty_or_whitespace_since_is_unfiltered(client, mock_storage) -> None:
    empty = client.get("/api/compliance/alerts?since=")
    assert empty.status_code == 200
    mock_storage.list_alerts.assert_awaited_with(
        policy_document_id=None,
        company_name=None,
        jurisdiction=None,
        since=None,
        limit=50,
        offset=0,
    )

    mock_storage.list_alerts.reset_mock()
    whitespace = client.get("/api/compliance/alerts?since=%20%20%20")
    assert whitespace.status_code == 200
    mock_storage.list_alerts.assert_awaited_once_with(
        policy_document_id=None,
        company_name=None,
        jurisdiction=None,
        since=None,
        limit=50,
        offset=0,
    )
