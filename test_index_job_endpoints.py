"""Tests for sub-vector index job endpoints in core.py."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


def test_create_sub_vector_index_requires_document_type(client, mock_clients) -> None:
    response = client.post("/create-sub-vector-index", json={"source_query": {"document_id": "doc-1"}})
    assert response.status_code == 400
    payload = response.get_json()
    assert "document_type is required" in payload["error"]


def test_create_sub_vector_index_rejects_invalid_source_query_string(client, mock_clients) -> None:
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "policy", "source_query": "{bad-json}"},
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert "source_query must be valid JSON" in payload["error"]


def test_create_sub_vector_index_normalizes_document_type_and_source_query(client, mock_clients) -> None:
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-123"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={
                "document_type": " Policy ",
                "source_query": '{"document_id":"doc-1"}',
            },
        )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload == {"job_id": "job-123", "status": "pending"}
    start_job.assert_called_once()
    call_kwargs = start_job.call_args.kwargs
    assert call_kwargs["request_dict"] == {
        "document_type": "policy",
        "source_query": {"document_id": "doc-1"},
    }
    assert call_kwargs["job_storage"] is mock_storage
    assert "flask_app" in call_kwargs


def test_create_sub_vector_index_non_dict_source_query_defaults_to_empty_dict(client, mock_clients) -> None:
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-abc"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": ["not", "a", "dict"]},
        )

    assert response.status_code == 202
    call_kwargs = start_job.call_args.kwargs
    assert call_kwargs["request_dict"] == {"document_type": "policy", "source_query": {}}


def test_create_sub_vector_index_returns_500_when_service_not_initialized(client, mock_clients) -> None:
    with patch(
        "core._get_index_job_storage",
        side_effect=RuntimeError("Index job service not initialized; call init_index_job first"),
    ):
        response = client.post("/create-sub-vector-index", json={"document_type": "policy"})

    assert response.status_code == 500
    payload = response.get_json()
    assert "Index job service not initialized" in payload["error"]


def test_get_index_job_returns_404_when_missing(client, mock_clients) -> None:
    mock_storage = MagicMock()
    mock_storage.get_job.return_value = None
    with patch("core._get_index_job_storage", return_value=mock_storage):
        response = client.get("/index-jobs/does-not-exist")

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["error"] == "Job not found"


def test_get_index_job_returns_job_payload(client, mock_clients) -> None:
    mock_storage = MagicMock()
    mock_storage.get_job.return_value = {
        "job_id": "job-123",
        "job_type": "sub_vector_index",
        "status": "completed",
        "result": {"status": "completed"},
    }
    with patch("core._get_index_job_storage", return_value=mock_storage):
        response = client.get("/index-jobs/job-123")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["job_id"] == "job-123"
    assert payload["status"] == "completed"
