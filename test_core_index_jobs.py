"""Tests for core index-jobs and create-sub-vector-index endpoints."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from compliance_config import load_config
from core import core_bp, init_core
from index_job_service import (
    DOCUMENT_TYPE_POLICY,
    DOCUMENT_TYPE_STATUTE,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_TYPE_SUB_VECTOR_INDEX,
    build_sub_vector_index_graph,
)


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


def _invoke_sub_vector_index_graph(app: Flask, request_dict: dict, job_id: str = "job-123"):
    storage = MagicMock()
    storage._config = load_config()
    graph = build_sub_vector_index_graph(app, storage)
    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": job_id,
                "job_type": JOB_TYPE_SUB_VECTOR_INDEX,
                "status": JOB_STATUS_PENDING,
                "request": request_dict,
            }
        )
    )
    return result, storage


def test_sub_vector_index_statute_graph_runs_expected_pipeline(app) -> None:
    source_query = {"document_id": "statute-1"}

    with patch("index_job_service._run_pipeline_step") as run_step:
        result, storage = _invoke_sub_vector_index_graph(
            app,
            {"document_type": DOCUMENT_TYPE_STATUTE, "source_query": source_query},
            job_id="job-statute",
        )

    assert result["status"] == JOB_STATUS_COMPLETED
    calls = [(call.args[1], call.args[2], call.args[3]) for call in run_step.call_args_list]
    assert [path for _, path, _ in calls] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]
    assert calls[0][2] == {
        "column": "chunk_text",
        "database": "privacy-compliance",
        "destination_collection": "statute_sub_chunks",
        "parse_prompt": "",
        "source_collection": "statute_chunks",
        "source_query": source_query,
        "subsection_column": "sub_chunk_text",
    }
    assert calls[1][2]["source_collection"] == "statute_sub_chunks"
    assert calls[1][2]["destination_collection"] == "statute_subtopics"
    assert calls[2][2] == {
        "index_collection_name": "statute_sub_embeddings",
        "index_database_name": "privacy-compliance",
        "source_collection_name": "statute_sub_chunks",
        "source_database_name": "privacy-compliance",
        "source_query": source_query,
        "text_column": "sub_chunk_text",
    }
    assert calls[3][2] == {
        "collection_name": "statute_sub_embeddings",
        "database_name": "privacy-compliance",
        "filter_fields": ["jurisdiction", "document_id"],
        "index_name": "vector_index",
    }
    storage.update_job_status.assert_any_call("job-statute", JOB_STATUS_RUNNING)
    storage.update_job_status.assert_any_call(
        "job-statute",
        JOB_STATUS_COMPLETED,
        result={
            "document_type": DOCUMENT_TYPE_STATUTE,
            "source_query": source_query,
            "status": "completed",
        },
    )


def test_sub_vector_index_policy_graph_uses_policy_embedding_pipeline(app) -> None:
    source_query = {"document_id": "policy-1"}

    with patch("index_job_service._run_pipeline_step") as run_step:
        result, storage = _invoke_sub_vector_index_graph(
            app,
            {"document_type": DOCUMENT_TYPE_POLICY, "source_query": source_query},
            job_id="job-policy",
        )

    assert result["status"] == JOB_STATUS_COMPLETED
    calls = [(call.args[1], call.args[2], call.args[3]) for call in run_step.call_args_list]
    assert [path for _, path, _ in calls] == [
        "/create-embeddings",
        "/create-vector-index",
    ]
    assert calls[0][2] == {
        "source_database_name": "privacy-compliance",
        "source_collection_name": "policy_chunks",
        "index_database_name": "privacy-compliance",
        "index_collection_name": "policy_legal_embeddings",
        "source_query": source_query,
        "text_column": "chunk_text",
    }
    assert calls[1][2] == {
        "collection_name": "policy_legal_embeddings",
        "database_name": "privacy-compliance",
        "filter_fields": ["document_id"],
        "index_name": "vector_index",
    }
    storage.update_job_status.assert_any_call("job-policy", JOB_STATUS_RUNNING)
    storage.update_job_status.assert_any_call(
        "job-policy",
        JOB_STATUS_COMPLETED,
        result={
            "document_type": DOCUMENT_TYPE_POLICY,
            "source_query": source_query,
            "status": "completed",
        },
    )


def test_sub_vector_index_graph_marks_job_failed_on_pipeline_error(app) -> None:
    with patch(
        "index_job_service._run_pipeline_step",
        side_effect=[None, RuntimeError("create-statute-subtopics failed")],
    ):
        result, storage = _invoke_sub_vector_index_graph(
            app,
            {"document_type": DOCUMENT_TYPE_STATUTE, "source_query": {}},
            job_id="job-failed",
        )

    assert result["status"] == JOB_STATUS_FAILED
    assert "create-statute-subtopics failed" in result["error"]
    storage.update_job_status.assert_any_call("job-failed", JOB_STATUS_RUNNING)
    storage.update_job_status.assert_any_call(
        "job-failed",
        JOB_STATUS_FAILED,
        error="create-statute-subtopics failed",
    )
