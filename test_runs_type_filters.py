"""GET /runs type and policy-id filters.

Empty and whitespace date filters are covered elsewhere. These tests pin
how the types list is tokenized and how a blank policy id is passed through
before Mongo is queried.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp


@pytest.fixture
def client():
    config = load_config()
    config.auth_required = False
    suite = MagicMock()
    suite._storage = MagicMock()
    suite._storage.list_runs = AsyncMock(return_value=([], 0))
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=suite
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client(), suite


def _kwargs(suite: MagicMock) -> dict:
    return suite._storage.list_runs.await_args.kwargs


def test_runs_types_strip_tokens_and_drop_blanks(client) -> None:
    http, suite = client
    response = http.get("/api/compliance/runs?types=%20gap%20,%20,%20health_score%20")

    assert response.status_code == 200
    assert _kwargs(suite) == {
        "policy_document_id": None,
        "since": None,
        "until": None,
        "types": ["gap", "health_score"],
        "limit": 50,
        "offset": 0,
    }


def test_runs_types_of_only_separators_is_unfiltered(client) -> None:
    http, suite = client
    response = http.get("/api/compliance/runs?types=%20,%20")

    assert response.status_code == 200
    assert _kwargs(suite)["types"] is None


def test_runs_blank_policy_id_is_unfiltered_and_whitespace_is_kept(client) -> None:
    http, suite = client

    empty = http.get("/api/compliance/runs?policy_document_id=")
    assert empty.status_code == 200
    assert _kwargs(suite)["policy_document_id"] is None

    padded = http.get("/api/compliance/runs?policy_document_id=%20%20")
    assert padded.status_code == 200
    assert _kwargs(suite)["policy_document_id"] == "  "
