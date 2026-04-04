"""Regression tests for sub-vector-index pipeline orchestration."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from flask import Flask, jsonify, request

from index_job_service import (
    DOCUMENT_TYPE_POLICY,
    DOCUMENT_TYPE_STATUTE,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    build_sub_vector_index_graph,
)


def test_policy_pipeline_success_runs_expected_steps_and_updates_job() -> None:
    """Policy jobs should run embeddings then vector index and complete."""
    app = Flask(__name__)
    calls: list[tuple[str, dict]] = []

    @app.post("/create-embeddings")
    def create_embeddings():
        payload = request.get_json() or {}
        calls.append(("create-embeddings", payload))
        return jsonify({"ok": True}), 200

    @app.post("/create-vector-index")
    def create_vector_index():
        payload = request.get_json() or {}
        calls.append(("create-vector-index", payload))
        return jsonify({"ok": True}), 200

    config = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks",
        policy_legal_embeddings_collection="policy_legal_embeddings",
    )
    storage = SimpleNamespace(_config=config, update_job_status=MagicMock())
    graph = build_sub_vector_index_graph(app, storage)

    source_query = {"document_id": "policy-123"}
    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-policy",
                "request": {"document_type": DOCUMENT_TYPE_POLICY, "source_query": source_query},
            }
        )
    )

    assert [name for name, _ in calls] == ["create-embeddings", "create-vector-index"]
    assert calls[0][1] == {
        "source_database_name": "privacy-compliance",
        "source_collection_name": "policy_chunks",
        "index_database_name": "privacy-compliance",
        "index_collection_name": "policy_legal_embeddings",
        "source_query": {"document_id": "policy-123"},
        "text_column": "chunk_text",
    }
    assert calls[1][1] == {
        "collection_name": "policy_legal_embeddings",
        "database_name": "privacy-compliance",
        "filter_fields": ["document_id"],
        "index_name": "vector_index",
    }

    expected_result = {
        "document_type": DOCUMENT_TYPE_POLICY,
        "source_query": {"document_id": "policy-123"},
        "status": "completed",
    }
    assert result["status"] == JOB_STATUS_COMPLETED
    assert result["result"] == expected_result
    storage.update_job_status.assert_has_calls(
        [
            call("job-policy", JOB_STATUS_RUNNING),
            call("job-policy", JOB_STATUS_COMPLETED, result=expected_result),
        ]
    )


def test_statute_pipeline_success_runs_all_four_steps_in_order() -> None:
    """Statute jobs should run subsections, subtopics, embeddings, then index."""
    app = Flask(__name__)
    calls: list[tuple[str, dict]] = []

    @app.post("/create-statute-subsections")
    def create_statute_subsections():
        payload = request.get_json() or {}
        calls.append(("create-statute-subsections", payload))
        return jsonify({"ok": True}), 200

    @app.post("/create-statute-subtopics")
    def create_statute_subtopics():
        payload = request.get_json() or {}
        calls.append(("create-statute-subtopics", payload))
        return jsonify({"ok": True}), 200

    @app.post("/create-embeddings")
    def create_embeddings():
        payload = request.get_json() or {}
        calls.append(("create-embeddings", payload))
        return jsonify({"ok": True}), 200

    @app.post("/create-vector-index")
    def create_vector_index():
        payload = request.get_json() or {}
        calls.append(("create-vector-index", payload))
        return jsonify({"ok": True}), 200

    config = SimpleNamespace(compliance_database="privacy-compliance")
    storage = SimpleNamespace(_config=config, update_job_status=MagicMock())
    graph = build_sub_vector_index_graph(app, storage)

    source_query = {"jurisdiction": "CA", "document_id": "statute-9"}
    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-statute",
                "request": {"document_type": DOCUMENT_TYPE_STATUTE, "source_query": source_query},
            }
        )
    )

    assert [name for name, _ in calls] == [
        "create-statute-subsections",
        "create-statute-subtopics",
        "create-embeddings",
        "create-vector-index",
    ]
    assert calls[0][1]["source_query"] == source_query
    assert calls[1][1]["source_query"] == source_query
    assert calls[2][1]["source_query"] == source_query
    assert result["status"] == JOB_STATUS_COMPLETED


def test_pipeline_step_failure_marks_job_failed_with_error() -> None:
    """A failing pipeline step should transition the job to failed."""
    app = Flask(__name__)
    calls: list[str] = []

    @app.post("/create-embeddings")
    def create_embeddings():
        calls.append("create-embeddings")
        return jsonify({"error": "embedding backend unavailable"}), 500

    @app.post("/create-vector-index")
    def create_vector_index():
        calls.append("create-vector-index")
        return jsonify({"ok": True}), 200

    config = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks",
        policy_legal_embeddings_collection="policy_legal_embeddings",
    )
    storage = SimpleNamespace(_config=config, update_job_status=MagicMock())
    graph = build_sub_vector_index_graph(app, storage)

    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-fail",
                "request": {"document_type": DOCUMENT_TYPE_POLICY, "source_query": {}},
            }
        )
    )

    assert calls == ["create-embeddings"]
    assert result["status"] == JOB_STATUS_FAILED
    assert "create-embeddings" in result["error"]
    assert "embedding backend unavailable" in result["error"]

    assert storage.update_job_status.call_args_list[0] == call("job-fail", JOB_STATUS_RUNNING)
    failed_call = storage.update_job_status.call_args_list[1]
    assert failed_call.args == ("job-fail", JOB_STATUS_FAILED)
    assert "create-embeddings" in failed_call.kwargs["error"]
