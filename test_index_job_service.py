"""Unit tests for sub-vector-index background job workflow."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from index_job_service import (
    DOCUMENT_TYPE_POLICY,
    DOCUMENT_TYPE_STATUTE,
    JOB_TYPE_SUB_VECTOR_INDEX,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    _run_pipeline_step,
    build_sub_vector_index_graph,
    start_sub_vector_index_job,
)


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks_custom",
        policy_legal_embeddings_collection="policy_legal_embeddings_custom",
    )


def _run_graph(graph, state):
    return asyncio.run(graph.ainvoke(state))


def test_build_graph_statute_pipeline_sequence() -> None:
    """Statute jobs must execute all expected pipeline steps in order."""
    app = Flask(__name__)
    job_storage = MagicMock()
    job_storage._config = _make_config()
    graph = build_sub_vector_index_graph(app, job_storage)

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _run_graph(
            graph,
            {
                "job_id": "job-statute-1",
                "request": {"document_type": DOCUMENT_TYPE_STATUTE, "source_query": {"doc": "a"}},
            },
        )

    assert result["status"] == JOB_STATUS_COMPLETED
    assert [
        (call.args[1], call.args[2]) for call in run_step.call_args_list
    ] == [
        ("create-statute-subsections", "/create-statute-subsections"),
        ("create-statute-subtopics", "/create-statute-subtopics"),
        ("create-embeddings", "/create-embeddings"),
        ("create-vector-index", "/create-vector-index"),
    ]

    subtopics_payload = run_step.call_args_list[1].args[3]
    assert subtopics_payload["source_collection"] == "statute_sub_chunks"
    assert subtopics_payload["destination_collection"] == "statute_subtopics"

    assert job_storage.update_job_status.call_args_list[0].args[:2] == (
        "job-statute-1",
        JOB_STATUS_RUNNING,
    )
    assert job_storage.update_job_status.call_args_list[-1].args[:2] == (
        "job-statute-1",
        JOB_STATUS_COMPLETED,
    )


def test_build_graph_policy_uses_configured_collections() -> None:
    """Policy jobs must respect configured source/index collection names."""
    app = Flask(__name__)
    job_storage = MagicMock()
    job_storage._config = _make_config()
    graph = build_sub_vector_index_graph(app, job_storage)

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _run_graph(
            graph,
            {
                "job_id": "job-policy-1",
                "request": {"document_type": DOCUMENT_TYPE_POLICY, "source_query": {"document_id": "p1"}},
            },
        )

    assert result["status"] == JOB_STATUS_COMPLETED
    assert len(run_step.call_args_list) == 2
    embeddings_payload = run_step.call_args_list[0].args[3]
    vector_payload = run_step.call_args_list[1].args[3]
    assert embeddings_payload["source_collection_name"] == "policy_chunks_custom"
    assert embeddings_payload["index_collection_name"] == "policy_legal_embeddings_custom"
    assert vector_payload["collection_name"] == "policy_legal_embeddings_custom"


def test_build_graph_invalid_document_type_marks_job_failed() -> None:
    """Invalid document_type should fail the job and skip pipeline calls."""
    app = Flask(__name__)
    job_storage = MagicMock()
    job_storage._config = _make_config()
    graph = build_sub_vector_index_graph(app, job_storage)

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _run_graph(
            graph,
            {
                "job_id": "job-invalid-1",
                "request": {"document_type": "invalid"},
            },
        )

    assert result["status"] == JOB_STATUS_FAILED
    assert "Invalid document_type" in result["error"]
    run_step.assert_not_called()
    assert job_storage.update_job_status.call_args_list[-1].args[:2] == (
        "job-invalid-1",
        JOB_STATUS_FAILED,
    )


def test_run_pipeline_step_raises_runtime_error_with_error_message() -> None:
    """_run_pipeline_step should surface endpoint error details."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 500
    response.get_json.return_value = {"error": "bad payload"}
    response.get_data.return_value = b"fallback"
    client.post.return_value = response

    with pytest.raises(RuntimeError, match=r"POST /test-path failed \(500\): bad payload"):
        _run_pipeline_step(client, "POST", "/test-path", {"k": "v"})


def test_start_sub_vector_index_job_creates_background_thread() -> None:
    """start_sub_vector_index_job should create and start a daemon thread."""
    app = Flask(__name__)
    job_storage = MagicMock()
    job_storage.create_job.return_value = "job-thread-1"
    graph = MagicMock()

    with patch("index_job_service.build_sub_vector_index_graph", return_value=graph), patch(
        "index_job_service.threading.Thread"
    ) as thread_cls:
        thread_instance = thread_cls.return_value
        job_id = start_sub_vector_index_job(
            request_dict={"document_type": DOCUMENT_TYPE_POLICY, "source_query": {}},
            job_storage=job_storage,
            flask_app=app,
        )

    assert job_id == "job-thread-1"
    job_storage.create_job.assert_called_once_with(
        JOB_TYPE_SUB_VECTOR_INDEX,
        {"document_type": DOCUMENT_TYPE_POLICY, "source_query": {}},
    )
    thread_cls.assert_called_once()
    assert thread_cls.call_args.kwargs["daemon"] is True
    thread_instance.start.assert_called_once()
