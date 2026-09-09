"""POST /create-sub-vector-index maps uninitialized storage to JSON 500."""
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

UNINITIALIZED_MSG = "Index job service not initialized; call init_index_job first"


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


def test_create_sub_vector_index_uninitialized_returns_json_500(client) -> None:
    """Unlike GET /index-jobs/<id>, POST catches RuntimeError and returns JSON 500."""
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "policy", "source_query": {}},
    )
    assert response.status_code == 500
    payload = response.get_json()
    assert payload == {"error": UNINITIALIZED_MSG}


def test_create_sub_vector_index_uninitialized_statute_also_json_500(client) -> None:
    """The JSON 500 mapping is independent of document_type once validation passes."""
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "statute"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == UNINITIALIZED_MSG
