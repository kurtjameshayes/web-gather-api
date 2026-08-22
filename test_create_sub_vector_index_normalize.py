"""Regression tests for /create-sub-vector-index payload normalization."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

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


def test_create_sub_vector_index_strips_and_lowercases_document_type(client) -> None:
    """Swagger/copied values like ' Statute ' must normalize before the job starts."""
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-norm-1"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "  Statute  ", "source_query": {"jurisdiction": "CA"}},
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-norm-1"
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["document_type"] == "statute"
    assert request_dict["source_query"] == {"jurisdiction": "CA"}


def test_create_sub_vector_index_parses_source_query_json_string(client) -> None:
    """source_query may arrive as a JSON string from form/Swagger clients."""
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-norm-2"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={
                "document_type": "policy",
                "source_query": '{"document_id":"pol-1"}',
            },
        )

    assert response.status_code == 202
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["document_type"] == "policy"
    assert request_dict["source_query"] == {"document_id": "pol-1"}


def test_create_sub_vector_index_non_object_source_query_becomes_empty_dict(client) -> None:
    """Non-object source_query values are coerced to {} rather than rejected."""
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-norm-3"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": ["not", "object"]},
        )

    assert response.status_code == 202
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["source_query"] == {}
