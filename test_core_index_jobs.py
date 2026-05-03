"""Tests for core index-jobs and create-sub-vector-index endpoints."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from compliance_config import load_config
from core import core_bp, init_core
from index_job_service import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
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


@pytest.mark.anyio
async def test_sub_vector_index_graph_policy_uses_legal_embedding_pipeline() -> None:
    """Policy indexing should skip statute subchunking and use configured legal embedding collections."""
    app = Flask(__name__)
    config = load_config()
    job_storage = MagicMock()
    job_storage._config = config
    calls = []

    with patch("index_job_service._run_pipeline_step", side_effect=lambda *args: calls.append(args[1:])):
        graph = build_sub_vector_index_graph(app, job_storage)
        result = await graph.ainvoke({
            "job_id": "idx-policy-1",
            "request": {
                "document_type": "policy",
                "source_query": {"document_id": "policy-1"},
            },
        })

    assert result["status"] == JOB_STATUS_COMPLETED
    assert [call[0] for call in calls] == ["create-embeddings", "create-vector-index"]
    assert [call[1] for call in calls] == ["/create-embeddings", "/create-vector-index"]
    assert calls[0][2] == {
        "source_database_name": config.compliance_database,
        "source_collection_name": config.policy_chunks_collection,
        "index_database_name": config.compliance_database,
        "index_collection_name": config.policy_legal_embeddings_collection,
        "source_query": {"document_id": "policy-1"},
        "text_column": "chunk_text",
    }
    assert calls[1][2] == {
        "collection_name": config.policy_legal_embeddings_collection,
        "database_name": config.compliance_database,
        "filter_fields": ["document_id"],
        "index_name": "vector_index",
    }
    job_storage.update_job_status.assert_any_call("idx-policy-1", JOB_STATUS_RUNNING)
    job_storage.update_job_status.assert_any_call(
        "idx-policy-1",
        JOB_STATUS_COMPLETED,
        result={
            "document_type": "policy",
            "source_query": {"document_id": "policy-1"},
            "status": "completed",
        },
    )


@pytest.mark.anyio
async def test_sub_vector_index_graph_statute_runs_full_subchunk_pipeline() -> None:
    """Statute indexing should execute all four pipeline steps in the required order."""
    app = Flask(__name__)
    config = load_config()
    job_storage = MagicMock()
    job_storage._config = config
    calls = []

    with patch("index_job_service._run_pipeline_step", side_effect=lambda *args: calls.append(args[1:])):
        graph = build_sub_vector_index_graph(app, job_storage)
        result = await graph.ainvoke({
            "job_id": "idx-statute-1",
            "request": {
                "document_type": "statute",
                "source_query": {"jurisdiction": "CA"},
            },
        })

    assert result["status"] == JOB_STATUS_COMPLETED
    assert [call[0] for call in calls] == [
        "create-statute-subsections",
        "create-statute-subtopics",
        "create-embeddings",
        "create-vector-index",
    ]
    assert calls[0][1] == "/create-statute-subsections"
    assert calls[0][2]["destination_collection"] == "statute_sub_chunks"
    assert calls[0][2]["source_collection"] == "statute_chunks"
    assert calls[1][1] == "/create-statute-subtopics"
    assert calls[1][2]["destination_collection"] == "statute_subtopics"
    assert calls[1][2]["source_collection"] == "statute_sub_chunks"
    assert calls[2][1] == "/create-embeddings"
    assert calls[2][2]["source_collection_name"] == "statute_sub_chunks"
    assert calls[2][2]["index_collection_name"] == "statute_sub_embeddings"
    assert calls[2][2]["text_column"] == "sub_chunk_text"
    assert calls[3][1] == "/create-vector-index"
    assert calls[3][2]["collection_name"] == "statute_sub_embeddings"
    assert calls[3][2]["filter_fields"] == ["jurisdiction", "document_id"]


def test_run_pipeline_step_reports_json_error_body() -> None:
    """Pipeline failures should expose endpoint error details in the job error."""
    from index_job_service import _run_pipeline_step

    response = SimpleNamespace(
        status_code=422,
        get_json=lambda: {"error": "bad source query"},
        get_data=lambda as_text=False: "fallback body",
    )
    client = MagicMock()
    client.post.return_value = response

    with pytest.raises(RuntimeError) as exc:
        _run_pipeline_step(client, "create-embeddings", "/create-embeddings", {"source_query": {}})

    assert str(exc.value) == "create-embeddings /create-embeddings failed (422): bad source query"
    client.post.assert_called_once_with("/create-embeddings", json={"source_query": {}})
