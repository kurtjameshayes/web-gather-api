"""Unit tests for compliance_job_service, index_job_service, embedder."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

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
    JOB_STATUS_COMPLETED as INDEX_JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED as INDEX_JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING as INDEX_JOB_STATUS_RUNNING,
    IndexJobStorage,
    JOB_STATUS_PENDING,
    _run_pipeline_step,
    build_sub_vector_index_graph,
)


def _mock_response(status_code: int, json_body=None, text_body: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.get_json.return_value = json_body
    response.get_data.return_value = text_body
    return response


class _FakeFlaskApp:
    def __init__(self, client: MagicMock) -> None:
        self._client = client

    def app_context(self):
        return nullcontext()

    def test_client(self) -> MagicMock:
        return self._client


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
    assert doc["status"] == JOB_STATUS_PENDING

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


def test_run_pipeline_step_raises_on_json_error_response() -> None:
    """_run_pipeline_step raises RuntimeError with JSON error details."""
    client = MagicMock()
    client.post.return_value = _mock_response(422, json_body={"error": "invalid payload"})

    with pytest.raises(
        RuntimeError,
        match=r"create-embeddings /create-embeddings failed \(422\): invalid payload",
    ):
        _run_pipeline_step(client, "create-embeddings", "/create-embeddings", {"x": 1})


@pytest.mark.anyio
async def test_build_sub_vector_index_graph_policy_success_sequence() -> None:
    """Policy workflow runs embeddings then vector index and marks completed."""
    client = MagicMock()
    client.post.side_effect = [
        _mock_response(200, json_body={}),
        _mock_response(200, json_body={}),
    ]
    job_storage = MagicMock()
    job_storage._config = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks_custom",
        policy_legal_embeddings_collection="policy_legal_embeddings_custom",
    )
    graph = build_sub_vector_index_graph(_FakeFlaskApp(client), job_storage)

    final_state = await graph.ainvoke(
        {
            "job_id": "job-policy-1",
            "status": JOB_STATUS_PENDING,
            "request": {
                "document_type": DOCUMENT_TYPE_POLICY,
                "source_query": {"document_id": "doc-123"},
            },
        }
    )

    assert final_state["status"] == INDEX_JOB_STATUS_COMPLETED
    assert final_state["result"]["document_type"] == DOCUMENT_TYPE_POLICY
    assert final_state["result"]["source_query"] == {"document_id": "doc-123"}
    assert client.post.call_args_list == [
        call(
            "/create-embeddings",
            json={
                "source_database_name": "privacy-compliance",
                "source_collection_name": "policy_chunks_custom",
                "index_database_name": "privacy-compliance",
                "index_collection_name": "policy_legal_embeddings_custom",
                "source_query": {"document_id": "doc-123"},
                "text_column": "chunk_text",
            },
        ),
        call(
            "/create-vector-index",
            json={
                "collection_name": "policy_legal_embeddings_custom",
                "database_name": "privacy-compliance",
                "filter_fields": ["document_id"],
                "index_name": "vector_index",
            },
        ),
    ]
    assert job_storage.update_job_status.call_args_list[0] == call(
        "job-policy-1",
        INDEX_JOB_STATUS_RUNNING,
    )
    completed_call = job_storage.update_job_status.call_args_list[1]
    assert completed_call.args == ("job-policy-1", INDEX_JOB_STATUS_COMPLETED)
    assert completed_call.kwargs["result"]["document_type"] == DOCUMENT_TYPE_POLICY


@pytest.mark.anyio
async def test_build_sub_vector_index_graph_statute_success_sequence() -> None:
    """Statute workflow runs all expected endpoints in order."""
    client = MagicMock()
    client.post.side_effect = [
        _mock_response(200, json_body={}),
        _mock_response(200, json_body={}),
        _mock_response(200, json_body={}),
        _mock_response(200, json_body={}),
    ]
    job_storage = MagicMock()
    job_storage._config = SimpleNamespace(compliance_database="privacy-compliance")
    graph = build_sub_vector_index_graph(_FakeFlaskApp(client), job_storage)

    final_state = await graph.ainvoke(
        {
            "job_id": "job-statute-1",
            "status": JOB_STATUS_PENDING,
            "request": {
                "document_type": DOCUMENT_TYPE_STATUTE,
                "source_query": {"document_id": "stat-42"},
            },
        }
    )

    assert final_state["status"] == INDEX_JOB_STATUS_COMPLETED
    assert [c.args[0] for c in client.post.call_args_list] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]
    first_payload = client.post.call_args_list[0].kwargs["json"]
    assert first_payload["source_query"] == {"document_id": "stat-42"}
    assert first_payload["destination_collection"] == "statute_sub_chunks"
    second_payload = client.post.call_args_list[1].kwargs["json"]
    assert second_payload["destination_collection"] == "statute_subtopics"
    third_payload = client.post.call_args_list[2].kwargs["json"]
    assert third_payload["source_collection_name"] == "statute_sub_chunks"
    fourth_payload = client.post.call_args_list[3].kwargs["json"]
    assert fourth_payload["filter_fields"] == ["jurisdiction", "document_id"]


@pytest.mark.anyio
async def test_build_sub_vector_index_graph_invalid_document_type_fails_job() -> None:
    """Invalid document_type marks the job as failed without endpoint calls."""
    client = MagicMock()
    job_storage = MagicMock()
    job_storage._config = SimpleNamespace(compliance_database="privacy-compliance")
    graph = build_sub_vector_index_graph(_FakeFlaskApp(client), job_storage)

    final_state = await graph.ainvoke(
        {
            "job_id": "job-invalid-1",
            "status": JOB_STATUS_PENDING,
            "request": {"document_type": "unknown"},
        }
    )

    assert final_state["status"] == INDEX_JOB_STATUS_FAILED
    assert "Invalid document_type: unknown" in final_state["error"]
    client.post.assert_not_called()
    assert job_storage.update_job_status.call_args_list[0] == call(
        "job-invalid-1",
        INDEX_JOB_STATUS_RUNNING,
    )
    failed_call = job_storage.update_job_status.call_args_list[1]
    assert failed_call.args == ("job-invalid-1", INDEX_JOB_STATUS_FAILED)
    assert "Invalid document_type: unknown" in failed_call.kwargs["error"]


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
