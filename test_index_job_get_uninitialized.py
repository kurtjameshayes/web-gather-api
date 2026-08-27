"""GET /index-jobs/<job_id> does not catch uninitialized storage RuntimeError."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

import core
from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _restore_index_job_storage():
    previous = core._index_job_storage
    core._index_job_storage = None
    yield
    core._index_job_storage = previous


def test_get_index_job_uninitialized_returns_generic_500(client) -> None:
    """Unlike POST /create-sub-vector-index, GET does not map RuntimeError to JSON 500."""
    response = client.get("/index-jobs/job-123")
    assert response.status_code == 500
    payload = response.get_json(silent=True)
    assert payload is None or payload.get("error") != (
        "Index job service not initialized; call init_index_job first"
    )
