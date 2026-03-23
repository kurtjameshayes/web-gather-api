"""Tests for core index-jobs and create-sub-vector-index endpoints."""
from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest
from flask import Flask

from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    """Mock MongoDB, Firecrawl, Anthropic for core."""
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    """Flask app with core blueprint."""
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    """Test client."""
    return app.test_client()


def test_get_index_job_found(client) -> None:
    """GET /index-jobs/<job_id> returns job when found."""
    mock_storage = MagicMock()
    mock_storage.get_job.return_value = {
        "job_id": "job-123",
        "status": "completed",
        "result": {"document_type": "policy"},
    }
    with patch("core._get_index_job_storage", return_value=mock_storage):
        response = client.get("/index-jobs/job-123")
    assert response.status_code == 200
    data = response.get_json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "completed"


def test_get_index_job_not_found(client) -> None:
    """GET /index-jobs/<job_id> returns 404 when job not found."""
    mock_storage = MagicMock()
    mock_storage.get_job.return_value = None
    with patch("core._get_index_job_storage", return_value=mock_storage):
        response = client.get("/index-jobs/nonexistent-job")
    assert response.status_code == 404
    assert "not found" in response.get_json()["error"].lower()


def test_create_sub_vector_index_missing_document_type(client) -> None:
    """POST /create-sub-vector-index without document_type returns 400."""
    response = client.post("/create-sub-vector-index", json={})
    assert response.status_code == 400
    assert "document_type" in response.get_json()["error"].lower()


def test_create_sub_vector_index_invalid_document_type(client) -> None:
    """POST /create-sub-vector-index with invalid document_type returns 400."""
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "invalid"},
    )
    assert response.status_code == 400
    assert "policy" in response.get_json()["error"].lower() or "statute" in response.get_json()["error"].lower()


def test_create_sub_vector_index_success(client) -> None:
    """POST /create-sub-vector-index with valid payload returns 202 and job_id."""
    mock_storage = MagicMock()
    mock_storage.create_job.return_value = "job-uuid-123"
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-uuid-123"
    ):
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": {}},
        )
    assert response.status_code == 202
    data = response.get_json()
    assert data["job_id"] == "job-uuid-123"
    assert data["status"] == "pending"


def test_create_sub_vector_index_invalid_source_query_json(client) -> None:
    """POST /create-sub-vector-index with invalid source_query string returns 400."""
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "policy", "source_query": "{not valid json"},
    )
    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"].lower() or "json" in response.get_json()["error"].lower()


def test_create_sub_vector_index_invalid_body_json(client) -> None:
    """POST /create-sub-vector-index with invalid JSON body returns 400."""
    response = client.post(
        "/create-sub-vector-index",
        data="{bad json",
        content_type="application/json",
    )
    assert response.status_code == 400


def test_create_sub_vector_index_accepts_single_quoted_payload(client) -> None:
    """POST /create-sub-vector-index accepts single-quoted payload fallback parser."""
    mock_storage = MagicMock()
    mock_storage.create_job.return_value = "job-quoted-123"
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-quoted-123"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            data="{'document_type': 'Policy', 'source_query': {'document_id': 'doc-1'}}",
            content_type="application/json",
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-quoted-123"
    start_job.assert_called_once_with(
        request_dict={"document_type": "policy", "source_query": {"document_id": "doc-1"}},
        job_storage=mock_storage,
        flask_app=ANY,
    )


def test_create_sub_vector_index_requires_api_key_when_configured(client, monkeypatch) -> None:
    """POST /create-sub-vector-index returns 401 when APP_API_KEY is configured but missing."""
    monkeypatch.setattr("security._app_api_key", "secret-key")
    response = client.post("/create-sub-vector-index", json={"document_type": "policy"})
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."


def test_create_sub_vector_index_rejects_invalid_api_key_when_configured(client, monkeypatch) -> None:
    """POST /create-sub-vector-index returns 403 for wrong API key when configured."""
    monkeypatch.setattr("security._app_api_key", "secret-key")
    response = client.post(
        "/create-sub-vector-index",
        json={"document_type": "policy"},
        headers={"x-api-key": "wrong"},
    )
    assert response.status_code == 403
    assert response.get_json()["error"] == "Invalid API key."


def test_get_index_job_requires_api_key_when_configured(client, monkeypatch) -> None:
    """GET /index-jobs/<job_id> returns 401 when APP_API_KEY is configured but missing."""
    monkeypatch.setattr("security._app_api_key", "secret-key")
    response = client.get("/index-jobs/job-123")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Missing API key."
