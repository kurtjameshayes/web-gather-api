"""Tests for core index-jobs and create-sub-vector-index endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def test_create_sub_vector_index_rejects_dangerous_source_query(client) -> None:
    """Dangerous source_query operators must be rejected before job creation."""
    with patch("core._get_index_job_storage") as mock_get_storage, patch(
        "index_job_service.start_sub_vector_index_job"
    ) as mock_start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": {"$where": "sleep(1000)"}},
        )

    assert response.status_code == 400
    assert "$where" in response.get_json()["error"]
    mock_get_storage.assert_not_called()
    mock_start_job.assert_not_called()


def test_create_sub_vector_index_rejects_non_object_source_query(client) -> None:
    """source_query strings for index jobs must decode to JSON objects."""
    with patch("core._get_index_job_storage") as mock_get_storage:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": "[]"},
        )

    assert response.status_code == 400
    assert "json object" in response.get_json()["error"].lower()
    mock_get_storage.assert_not_called()


def test_create_sub_vector_index_invalid_body_json(client) -> None:
    """POST /create-sub-vector-index with invalid JSON body returns 400."""
    response = client.post(
        "/create-sub-vector-index",
        data="{bad json",
        content_type="application/json",
    )
    assert response.status_code == 400
