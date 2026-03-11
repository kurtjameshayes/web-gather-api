"""Regression tests for index background job pipelines."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

from index_job_service import (
    DOCUMENT_TYPE_POLICY,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_RUNNING,
    build_sub_vector_index_graph,
)


class DummyJobStorage:
    def __init__(self) -> None:
        self._config = SimpleNamespace(
            compliance_database="compliance_db",
            policy_chunks_collection="policy_chunks_custom",
            policy_legal_embeddings_collection="policy_legal_embeddings_custom",
        )
        self.status_updates = []

    def update_job_status(self, job_id, status, result=None, error=None) -> None:
        self.status_updates.append(
            {
                "job_id": job_id,
                "status": status,
                "result": result,
                "error": error,
            }
        )


def test_policy_pipeline_uses_configured_legal_embedding_collections() -> None:
    app = Flask(__name__)
    storage = DummyJobStorage()
    graph = build_sub_vector_index_graph(app, storage)
    initial_state = {
        "job_id": "job-123",
        "request": {
            "document_type": DOCUMENT_TYPE_POLICY,
            "source_query": {"document_id": "doc-1"},
        },
    }

    with patch("index_job_service._run_pipeline_step") as run_step:
        final_state = asyncio.run(graph.ainvoke(initial_state))

    assert final_state["status"] == JOB_STATUS_COMPLETED
    assert [update["status"] for update in storage.status_updates] == [
        JOB_STATUS_RUNNING,
        JOB_STATUS_COMPLETED,
    ]

    assert run_step.call_count == 2
    first_call = run_step.call_args_list[0].args
    second_call = run_step.call_args_list[1].args

    assert first_call[1:] == (
        "create-embeddings",
        "/create-embeddings",
        {
            "source_database_name": "compliance_db",
            "source_collection_name": "policy_chunks_custom",
            "index_database_name": "compliance_db",
            "index_collection_name": "policy_legal_embeddings_custom",
            "source_query": {"document_id": "doc-1"},
            "text_column": "chunk_text",
        },
    )
    assert second_call[1:] == (
        "create-vector-index",
        "/create-vector-index",
        {
            "collection_name": "policy_legal_embeddings_custom",
            "database_name": "compliance_db",
            "filter_fields": ["document_id"],
            "index_name": "vector_index",
        },
    )
    assert all(call.args[1] != "create-policy-subsections" for call in run_step.call_args_list)
