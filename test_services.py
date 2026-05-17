"""Unit tests for compliance_job_service, index_job_service, embedder."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask, jsonify, request

from compliance_config import load_config
from compliance_job_service import (
    ComplianceJobStorage,
    JOB_STATUS_COMPLETED as COMPLIANCE_JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED as COMPLIANCE_JOB_STATUS_FAILED,
    JOB_STATUS_PENDING as COMPLIANCE_JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING as COMPLIANCE_JOB_STATUS_RUNNING,
    start_gap_analysis_job,
)
from index_job_service import (
    DOCUMENT_TYPE_POLICY,
    DOCUMENT_TYPE_STATUTE,
    IndexJobStorage,
    JOB_STATUS_COMPLETED as INDEX_JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED as INDEX_JOB_STATUS_FAILED,
    JOB_STATUS_PENDING as INDEX_JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING as INDEX_JOB_STATUS_RUNNING,
    build_sub_vector_index_graph,
)


def _make_index_pipeline_app(failing_path: str | None = None):
    """Create a tiny Flask app that records index pipeline step calls."""
    app = Flask(__name__)
    calls = []
    paths = [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]

    for path in paths:
        endpoint = f"post_{path.strip('/').replace('-', '_')}"

        def handler(path=path):
            calls.append((path, request.get_json()))
            if path == failing_path:
                return jsonify({"error": "boom"}), 500
            return jsonify({"ok": True})

        app.add_url_rule(path, endpoint=endpoint, view_func=handler, methods=["POST"])

    return app, calls


def test_compliance_job_storage_create_and_get() -> None:
    """ComplianceJobStorage create_job returns job_id and get_job retrieves it."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = ComplianceJobStorage(mock_mongo, config)

    job_id = storage.create_job("gap_analysis", {"policy_document_id": "pol-1"})
    assert job_id is not None
    assert len(job_id) == 36  # UUID format
    mock_coll.insert_one.assert_called_once()
    doc = mock_coll.insert_one.call_args[0][0]
    assert doc["job_type"] == "gap_analysis"
    assert doc["status"] == COMPLIANCE_JOB_STATUS_PENDING
    assert doc["request"]["policy_document_id"] == "pol-1"

    mock_coll.find_one.return_value = {
        "job_id": job_id,
        "job_type": "gap_analysis",
        "status": "completed",
        "request": doc["request"],
    }
    retrieved = storage.get_job(job_id)
    assert retrieved is not None
    assert retrieved["job_id"] == job_id
    assert retrieved["status"] == "completed"
    assert "_id" not in retrieved


def test_compliance_job_storage_get_not_found() -> None:
    """ComplianceJobStorage get_job returns None when job not found."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.find_one.return_value = None
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = ComplianceJobStorage(mock_mongo, config)

    result = storage.get_job("nonexistent")
    assert result is None


def test_compliance_job_storage_update_status() -> None:
    """ComplianceJobStorage update_job_status updates the job."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = ComplianceJobStorage(mock_mongo, config)

    storage.update_job_status("job-1", COMPLIANCE_JOB_STATUS_RUNNING)
    mock_coll.update_one.assert_called_once()
    call_args = mock_coll.update_one.call_args[0]
    assert call_args[0] == {"job_id": "job-1"}
    assert "$set" in call_args[1]
    assert call_args[1]["$set"]["status"] == COMPLIANCE_JOB_STATUS_RUNNING
    assert "started_at" in call_args[1]["$set"]

    storage.update_job_status("job-1", COMPLIANCE_JOB_STATUS_COMPLETED, result={"gaps": []})
    assert mock_coll.update_one.call_count == 2
    call_args = mock_coll.update_one.call_args_list[1][0]
    assert call_args[1]["$set"]["status"] == COMPLIANCE_JOB_STATUS_COMPLETED
    assert call_args[1]["$set"]["result"] == {"gaps": []}
    assert "completed_at" in call_args[1]["$set"]


def test_compliance_job_storage_cleanup_zombie_jobs_marks_restart_failures() -> None:
    """Startup cleanup only fails pending/running jobs and records completion time."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.update_many.return_value.modified_count = 2
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = ComplianceJobStorage(mock_mongo, config)

    cleaned = storage.cleanup_zombie_jobs()

    assert cleaned == 2
    mock_coll.update_many.assert_called_once()
    query, update = mock_coll.update_many.call_args[0]
    assert query == {
        "status": {"$in": [COMPLIANCE_JOB_STATUS_PENDING, COMPLIANCE_JOB_STATUS_RUNNING]}
    }
    assert update["$set"]["status"] == COMPLIANCE_JOB_STATUS_FAILED
    assert update["$set"]["error"] == "Job interrupted by server restart."
    assert "completed_at" in update["$set"]


def test_index_job_storage_create_and_get() -> None:
    """IndexJobStorage create_job and get_job work correctly."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = IndexJobStorage(mock_mongo, config)

    job_id = storage.create_job("sub_vector_index", {"document_type": "policy"})
    assert job_id is not None
    mock_coll.insert_one.assert_called_once()
    doc = mock_coll.insert_one.call_args[0][0]
    assert doc["job_type"] == "sub_vector_index"
    assert doc["status"] == INDEX_JOB_STATUS_PENDING

    mock_coll.find_one.return_value = {
        "job_id": job_id,
        "job_type": "sub_vector_index",
        "status": "completed",
        "result": {"document_type": "policy"},
    }
    retrieved = storage.get_job(job_id)
    assert retrieved is not None
    assert retrieved["status"] == "completed"


def test_index_job_storage_get_not_found() -> None:
    """IndexJobStorage get_job returns None when not found."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.find_one.return_value = None
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = IndexJobStorage(mock_mongo, config)

    assert storage.get_job("nonexistent") is None


def test_sub_vector_index_policy_graph_uses_policy_legal_embeddings_pipeline() -> None:
    """Policy jobs embed policy chunks directly into the legal embeddings collection."""
    app, calls = _make_index_pipeline_app()
    config = load_config()
    job_storage = MagicMock()
    job_storage._config = config
    graph = build_sub_vector_index_graph(app, job_storage)
    source_query = {"document_id": "policy-123"}

    result = asyncio.run(graph.ainvoke({
        "job_id": "job-policy",
        "status": INDEX_JOB_STATUS_PENDING,
        "request": {
            "document_type": DOCUMENT_TYPE_POLICY,
            "source_query": source_query,
        },
    }))

    assert result["status"] == INDEX_JOB_STATUS_COMPLETED
    assert calls == [
        (
            "/create-embeddings",
            {
                "source_database_name": config.compliance_database,
                "source_collection_name": config.policy_chunks_collection,
                "index_database_name": config.compliance_database,
                "index_collection_name": config.policy_legal_embeddings_collection,
                "source_query": source_query,
                "text_column": "chunk_text",
            },
        ),
        (
            "/create-vector-index",
            {
                "collection_name": config.policy_legal_embeddings_collection,
                "database_name": config.compliance_database,
                "filter_fields": ["document_id"],
                "index_name": "vector_index",
            },
        ),
    ]
    job_storage.update_job_status.assert_any_call("job-policy", INDEX_JOB_STATUS_RUNNING)
    job_storage.update_job_status.assert_any_call(
        "job-policy",
        INDEX_JOB_STATUS_COMPLETED,
        result={
            "document_type": DOCUMENT_TYPE_POLICY,
            "source_query": source_query,
            "status": "completed",
        },
    )


def test_sub_vector_index_statute_graph_runs_all_steps_in_order() -> None:
    """Statute jobs must preserve the four-step subsection/subtopic/index pipeline."""
    app, calls = _make_index_pipeline_app()
    config = load_config()
    job_storage = MagicMock()
    job_storage._config = config
    graph = build_sub_vector_index_graph(app, job_storage)
    source_query = {"jurisdiction": "CA"}

    result = asyncio.run(graph.ainvoke({
        "job_id": "job-statute",
        "status": INDEX_JOB_STATUS_PENDING,
        "request": {
            "document_type": DOCUMENT_TYPE_STATUTE,
            "source_query": source_query,
        },
    }))

    assert result["status"] == INDEX_JOB_STATUS_COMPLETED
    assert [path for path, _payload in calls] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]
    assert calls[0][1] == {
        "column": "chunk_text",
        "database": config.compliance_database,
        "destination_collection": "statute_sub_chunks",
        "parse_prompt": "",
        "source_collection": "statute_chunks",
        "source_query": source_query,
        "subsection_column": "sub_chunk_text",
    }
    assert calls[1][1]["source_collection"] == "statute_sub_chunks"
    assert calls[1][1]["destination_collection"] == "statute_subtopics"
    assert calls[2][1] == {
        "index_collection_name": "statute_sub_embeddings",
        "index_database_name": config.compliance_database,
        "source_collection_name": "statute_sub_chunks",
        "source_database_name": config.compliance_database,
        "source_query": source_query,
        "text_column": "sub_chunk_text",
    }
    assert calls[3][1] == {
        "collection_name": "statute_sub_embeddings",
        "database_name": config.compliance_database,
        "filter_fields": ["jurisdiction", "document_id"],
        "index_name": "vector_index",
    }


def test_sub_vector_index_graph_marks_job_failed_when_pipeline_step_fails() -> None:
    """Pipeline endpoint failures should persist a failed job with the endpoint error."""
    app, calls = _make_index_pipeline_app(failing_path="/create-embeddings")
    config = load_config()
    job_storage = MagicMock()
    job_storage._config = config
    graph = build_sub_vector_index_graph(app, job_storage)

    result = asyncio.run(graph.ainvoke({
        "job_id": "job-failure",
        "status": INDEX_JOB_STATUS_PENDING,
        "request": {
            "document_type": DOCUMENT_TYPE_POLICY,
            "source_query": {},
        },
    }))

    assert result["status"] == INDEX_JOB_STATUS_FAILED
    assert result["error"] == "create-embeddings /create-embeddings failed (500): boom"
    assert calls == [("/create-embeddings", {
        "source_database_name": config.compliance_database,
        "source_collection_name": config.policy_chunks_collection,
        "index_database_name": config.compliance_database,
        "index_collection_name": config.policy_legal_embeddings_collection,
        "source_query": {},
        "text_column": "chunk_text",
    })]
    job_storage.update_job_status.assert_any_call("job-failure", INDEX_JOB_STATUS_RUNNING)
    job_storage.update_job_status.assert_any_call(
        "job-failure",
        INDEX_JOB_STATUS_FAILED,
        error="create-embeddings /create-embeddings failed (500): boom",
    )


@pytest.mark.anyio
async def test_embedder_cache_hit() -> None:
    """Embedder returns cached embed when available."""
    import sys
    sys.modules["sentence_transformers"] = MagicMock()
    from cache import SimpleLRUCache
    from embedder import Embedder

    cache = SimpleLRUCache(10, 60)
    cache.set("hash123", [0.1, 0.2, 0.3])
    mock_model = MagicMock()
    embedder = Embedder("test-model", cache)
    embedder._model = mock_model

    with patch("embedder.hash_text", return_value="hash123"):
        result = await embedder.embed("cached text")
    assert result == [0.1, 0.2, 0.3]
    mock_model.encode.assert_not_called()


@pytest.mark.anyio
async def test_embedder_empty_text() -> None:
    """Embedder returns empty list for empty text."""
    import sys
    sys.modules["sentence_transformers"] = MagicMock()
    from cache import SimpleLRUCache
    from embedder import Embedder

    cache = SimpleLRUCache(10, 60)
    embedder = Embedder("test-model", cache)
    result = await embedder.embed("")
    assert result == []
