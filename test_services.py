"""Unit tests for compliance_job_service, index_job_service, embedder."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask, jsonify, request

from compliance_config import load_config
from compliance_job_service import (
    ComplianceJobStorage,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
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


def _force_local_state_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the lightweight graph implementation for deterministic service tests."""
    import compliance_graph

    graph_module = types.ModuleType("langgraph.graph")
    graph_module.END = compliance_graph.END
    graph_module.START = compliance_graph.START
    graph_module.StateGraph = compliance_graph.StateGraph

    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.__path__ = []
    langgraph_module.graph = graph_module

    monkeypatch.setitem(sys.modules, "langgraph", langgraph_module)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)


def _pipeline_app(failures: dict[str, tuple[int, dict]] | None = None) -> tuple[Flask, list[dict]]:
    """Create stub pipeline endpoints and record every posted payload."""
    app = Flask(__name__)
    calls: list[dict] = []
    failures = failures or {}

    for path in (
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ):

        def handler(path: str = path):
            calls.append({"path": path, "json": request.get_json()})
            if path in failures:
                status_code, payload = failures[path]
                return jsonify(payload), status_code
            return jsonify({"ok": True})

        endpoint = path.strip("/").replace("-", "_")
        app.add_url_rule(path, endpoint, handler, methods=["POST"])

    return app, calls


def _mock_index_storage(config):
    storage = MagicMock()
    storage._config = config
    return storage


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
    assert doc["status"] == JOB_STATUS_PENDING
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

    storage.update_job_status("job-1", JOB_STATUS_RUNNING)
    mock_coll.update_one.assert_called_once()
    call_args = mock_coll.update_one.call_args[0]
    assert call_args[0] == {"job_id": "job-1"}
    assert "$set" in call_args[1]
    assert call_args[1]["$set"]["status"] == JOB_STATUS_RUNNING
    assert "started_at" in call_args[1]["$set"]

    storage.update_job_status("job-1", JOB_STATUS_COMPLETED, result={"gaps": []})
    assert mock_coll.update_one.call_count == 2
    call_args = mock_coll.update_one.call_args_list[1][0]
    assert call_args[1]["$set"]["status"] == JOB_STATUS_COMPLETED
    assert call_args[1]["$set"]["result"] == {"gaps": []}
    assert "completed_at" in call_args[1]["$set"]


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


@pytest.mark.anyio
async def test_sub_vector_index_policy_pipeline_completes_and_updates_job(monkeypatch) -> None:
    """Policy sub-vector jobs run embeddings and vector-index steps with policy payloads."""
    _force_local_state_graph(monkeypatch)
    config = load_config()
    storage = _mock_index_storage(config)
    app, calls = _pipeline_app()
    graph = build_sub_vector_index_graph(app, storage)
    source_query = {"document_id": "policy-42"}

    result = await graph.ainvoke(
        {
            "job_id": "job-policy-1",
            "job_type": "sub_vector_index",
            "status": INDEX_JOB_STATUS_PENDING,
            "request": {
                "document_type": DOCUMENT_TYPE_POLICY,
                "source_query": source_query,
            },
        }
    )

    assert result["status"] == INDEX_JOB_STATUS_COMPLETED
    assert result["result"] == {
        "document_type": DOCUMENT_TYPE_POLICY,
        "source_query": source_query,
        "status": "completed",
    }
    assert [call["path"] for call in calls] == ["/create-embeddings", "/create-vector-index"]

    embeddings_payload = calls[0]["json"]
    assert embeddings_payload["source_database_name"] == config.compliance_database
    assert embeddings_payload["source_collection_name"] == config.policy_chunks_collection
    assert embeddings_payload["index_database_name"] == config.compliance_database
    assert embeddings_payload["index_collection_name"] == config.policy_legal_embeddings_collection
    assert embeddings_payload["source_query"] == source_query
    assert embeddings_payload["text_column"] == "chunk_text"

    vector_index_payload = calls[1]["json"]
    assert vector_index_payload["database_name"] == config.compliance_database
    assert vector_index_payload["collection_name"] == config.policy_legal_embeddings_collection
    assert vector_index_payload["filter_fields"] == ["document_id"]
    assert vector_index_payload["index_name"] == "vector_index"

    assert storage.update_job_status.call_args_list[0].args == ("job-policy-1", INDEX_JOB_STATUS_RUNNING)
    completed_call = storage.update_job_status.call_args_list[1]
    assert completed_call.args == ("job-policy-1", INDEX_JOB_STATUS_COMPLETED)
    assert completed_call.kwargs["result"] == result["result"]


@pytest.mark.anyio
async def test_sub_vector_index_statute_pipeline_calls_four_steps_in_order(monkeypatch) -> None:
    """Statute sub-vector jobs preserve subsection, subtopic, embedding, index order."""
    _force_local_state_graph(monkeypatch)
    config = load_config()
    storage = _mock_index_storage(config)
    app, calls = _pipeline_app()
    graph = build_sub_vector_index_graph(app, storage)
    source_query = {"jurisdiction": "CA", "document_id": "ccpa"}

    result = await graph.ainvoke(
        {
            "job_id": "job-statute-1",
            "job_type": "sub_vector_index",
            "status": INDEX_JOB_STATUS_PENDING,
            "request": {
                "document_type": DOCUMENT_TYPE_STATUTE,
                "source_query": source_query,
            },
        }
    )

    assert result["status"] == INDEX_JOB_STATUS_COMPLETED
    assert [call["path"] for call in calls] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]

    subsection_payload = calls[0]["json"]
    assert subsection_payload["database"] == config.compliance_database
    assert subsection_payload["source_collection"] == "statute_chunks"
    assert subsection_payload["destination_collection"] == "statute_sub_chunks"
    assert subsection_payload["source_query"] == source_query

    subtopic_payload = calls[1]["json"]
    assert subtopic_payload["database"] == config.compliance_database
    assert subtopic_payload["source_collection"] == "statute_sub_chunks"
    assert subtopic_payload["destination_collection"] == "statute_subtopics"
    assert subtopic_payload["source_query"] == source_query

    embeddings_payload = calls[2]["json"]
    assert embeddings_payload["source_database_name"] == config.compliance_database
    assert embeddings_payload["source_collection_name"] == "statute_sub_chunks"
    assert embeddings_payload["index_database_name"] == config.compliance_database
    assert embeddings_payload["index_collection_name"] == "statute_sub_embeddings"
    assert embeddings_payload["source_query"] == source_query
    assert embeddings_payload["text_column"] == "sub_chunk_text"

    vector_index_payload = calls[3]["json"]
    assert vector_index_payload["database_name"] == config.compliance_database
    assert vector_index_payload["collection_name"] == "statute_sub_embeddings"
    assert vector_index_payload["filter_fields"] == ["jurisdiction", "document_id"]
    assert vector_index_payload["index_name"] == "vector_index"


@pytest.mark.anyio
async def test_sub_vector_index_pipeline_marks_job_failed_on_step_error(monkeypatch) -> None:
    """Pipeline step failures are persisted so polling clients see failed jobs."""
    _force_local_state_graph(monkeypatch)
    config = load_config()
    storage = _mock_index_storage(config)
    app, calls = _pipeline_app(failures={"/create-vector-index": (400, {"error": "index rejected"})})
    graph = build_sub_vector_index_graph(app, storage)

    result = await graph.ainvoke(
        {
            "job_id": "job-policy-fail",
            "job_type": "sub_vector_index",
            "status": INDEX_JOB_STATUS_PENDING,
            "request": {
                "document_type": DOCUMENT_TYPE_POLICY,
                "source_query": {"document_id": "policy-42"},
            },
        }
    )

    assert result["status"] == INDEX_JOB_STATUS_FAILED
    assert "index rejected" in result["error"]
    assert [call["path"] for call in calls] == ["/create-embeddings", "/create-vector-index"]

    assert storage.update_job_status.call_args_list[0].args == ("job-policy-fail", INDEX_JOB_STATUS_RUNNING)
    failed_call = storage.update_job_status.call_args_list[1]
    assert failed_call.args == ("job-policy-fail", INDEX_JOB_STATUS_FAILED)
    assert "index rejected" in failed_call.kwargs["error"]


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
