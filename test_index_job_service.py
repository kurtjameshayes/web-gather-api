"""Tests for sub-vector-index background job graph behavior."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from flask import Flask

from compliance_config import load_config
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


def _job_storage() -> MagicMock:
    storage = MagicMock()
    storage._config = load_config()
    return storage


def _invoke_graph(app: Flask, storage: MagicMock, request: dict[str, object]) -> dict[str, object]:
    graph = build_sub_vector_index_graph(app, storage)
    return asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-1",
                "job_type": JOB_TYPE_SUB_VECTOR_INDEX,
                "status": JOB_STATUS_PENDING,
                "request": request,
            }
        )
    )


def test_sub_vector_index_graph_runs_policy_embedding_pipeline() -> None:
    """Policy jobs should embed policy chunks directly and then create the vector index."""
    app = Flask(__name__)
    storage = _job_storage()
    source_query = {"document_id": "policy-1"}

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _invoke_graph(
            app,
            storage,
            {"document_type": DOCUMENT_TYPE_POLICY, "source_query": source_query},
        )

    cfg = storage._config
    assert result["status"] == JOB_STATUS_COMPLETED
    assert result["result"] == {
        "document_type": DOCUMENT_TYPE_POLICY,
        "source_query": source_query,
        "status": "completed",
    }
    assert [call.args[1] for call in run_step.call_args_list] == [
        "create-embeddings",
        "create-vector-index",
    ]
    assert run_step.call_args_list[0].args[3] == {
        "source_database_name": cfg.compliance_database,
        "source_collection_name": cfg.policy_chunks_collection,
        "index_database_name": cfg.compliance_database,
        "index_collection_name": cfg.policy_legal_embeddings_collection,
        "source_query": source_query,
        "text_column": "chunk_text",
    }
    assert run_step.call_args_list[1].args[3] == {
        "collection_name": cfg.policy_legal_embeddings_collection,
        "database_name": cfg.compliance_database,
        "filter_fields": ["document_id"],
        "index_name": "vector_index",
    }
    storage.update_job_status.assert_any_call("job-1", JOB_STATUS_RUNNING)
    storage.update_job_status.assert_any_call("job-1", JOB_STATUS_COMPLETED, result=result["result"])


def test_sub_vector_index_graph_runs_statute_pipeline_in_order() -> None:
    """Statute jobs should preserve the full subsection, subtopic, embedding, index chain."""
    app = Flask(__name__)
    storage = _job_storage()
    source_query = {"jurisdiction": "CA", "document_id": "statute-1"}

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _invoke_graph(
            app,
            storage,
            {"document_type": DOCUMENT_TYPE_STATUTE, "source_query": source_query},
        )

    assert result["status"] == JOB_STATUS_COMPLETED
    assert [call.args[1] for call in run_step.call_args_list] == [
        "create-statute-subsections",
        "create-statute-subtopics",
        "create-embeddings",
        "create-vector-index",
    ]
    assert run_step.call_args_list[0].args[2] == "/create-statute-subsections"
    assert run_step.call_args_list[0].args[3]["source_collection"] == "statute_chunks"
    assert run_step.call_args_list[0].args[3]["destination_collection"] == "statute_sub_chunks"
    assert run_step.call_args_list[1].args[3]["destination_collection"] == "statute_subtopics"
    assert run_step.call_args_list[2].args[3]["index_collection_name"] == "statute_sub_embeddings"
    assert run_step.call_args_list[3].args[3]["filter_fields"] == ["jurisdiction", "document_id"]


def test_sub_vector_index_graph_marks_invalid_document_type_failed() -> None:
    """Invalid job payloads should fail the job instead of reporting completion."""
    app = Flask(__name__)
    storage = _job_storage()

    with patch("index_job_service._run_pipeline_step") as run_step:
        result = _invoke_graph(app, storage, {"document_type": "contract"})

    assert result["status"] == JOB_STATUS_FAILED
    assert "Invalid document_type: contract" == result["error"]
    run_step.assert_not_called()
    storage.update_job_status.assert_any_call("job-1", JOB_STATUS_RUNNING)
    storage.update_job_status.assert_any_call(
        "job-1",
        JOB_STATUS_FAILED,
        error="Invalid document_type: contract",
    )
