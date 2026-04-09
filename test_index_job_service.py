"""Regression tests for index job policy pipeline steps."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flask import Flask

from index_job_service import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_RUNNING,
    build_sub_vector_index_graph,
)


def test_policy_sub_vector_index_pipeline_uses_configured_collections() -> None:
    app = Flask(__name__)
    cfg = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks",
        policy_legal_embeddings_collection="policy_legal_embeddings",
    )

    job_storage = MagicMock()
    job_storage._config = cfg

    graph = build_sub_vector_index_graph(app, job_storage)
    initial_state = {
        "job_id": "job-123",
        "job_type": "sub_vector_index",
        "status": "pending",
        "request": {"document_type": "policy", "source_query": {"document_id": "pol-1"}},
    }

    with patch("index_job_service._run_pipeline_step") as run_step:
        final_state = asyncio.run(graph.ainvoke(initial_state))

    assert final_state["status"] == JOB_STATUS_COMPLETED

    assert run_step.call_count == 2
    first_call = run_step.call_args_list[0]
    second_call = run_step.call_args_list[1]

    assert first_call.args[1] == "create-embeddings"
    assert first_call.args[2] == "/create-embeddings"
    assert first_call.args[3]["source_database_name"] == "privacy-compliance"
    assert first_call.args[3]["source_collection_name"] == "policy_chunks"
    assert first_call.args[3]["index_collection_name"] == "policy_legal_embeddings"
    assert first_call.args[3]["text_column"] == "chunk_text"

    assert second_call.args[1] == "create-vector-index"
    assert second_call.args[2] == "/create-vector-index"
    assert second_call.args[3]["collection_name"] == "policy_legal_embeddings"
    assert second_call.args[3]["filter_fields"] == ["document_id"]

    called_paths = [c.args[2] for c in run_step.call_args_list]
    assert "/create-policy-subsections" not in called_paths

    job_storage.update_job_status.assert_any_call("job-123", JOB_STATUS_RUNNING)
    job_storage.update_job_status.assert_any_call(
        "job-123",
        JOB_STATUS_COMPLETED,
        result={
            "document_type": "policy",
            "source_query": {"document_id": "pol-1"},
            "status": "completed",
        },
    )
