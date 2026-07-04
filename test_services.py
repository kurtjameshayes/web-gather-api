"""Unit tests for compliance_job_service, index_job_service, embedder."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from compliance_config import load_config
from compliance_job_service import (
    ComplianceJobStorage,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    build_gap_analysis_graph,
    start_gap_analysis_job,
)
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    RetrievalMetadata,
)
from index_job_service import IndexJobStorage, JOB_STATUS_PENDING as INDEX_JOB_STATUS_PENDING


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


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gap_analysis_graph_persists_result_and_run_log(
    anyio_backend: str,
) -> None:
    """Completed gap jobs persist compliance results/run logs before marking completed."""
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    compliance_storage.write_compliance_run_log = AsyncMock()

    async def run_gap_analysis(request):
        assert request.policy_document_id == "pol-1"
        assert request.save_results is False
        return GapAnalysisResponse(
            policy_document_id="pol-1",
            company_name="ExampleCo",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-07-04T10:00:00Z",
            gaps=[
                GapItem(
                    jurisdiction="CA",
                    statute_reference="CCPA 1798.100",
                    statute_chunk_id="stat-1",
                    requirement_summary="Disclose categories of personal information collected.",
                    status="addressed",
                    policy_quote="We describe the categories of personal information we collect.",
                    confidence="high",
                )
            ],
            summary=GapSummary(total_requirements=1, addressed=1),
            retrieval_metadata=RetrievalMetadata(
                statute_items_considered=2,
                statute_pairs_matched=1,
            ),
            statute_chunk_ids_used=["stat-1"],
            run_types=["gap_v4"],
            run_type="gap_analysis_v4",
            version="v4",
        )

    with patch.dict("sys.modules", {"langgraph.graph": None}):
        graph = build_gap_analysis_graph(run_gap_analysis, job_storage, compliance_storage)

    state = await graph.ainvoke(
        {
            "job_id": "job-1",
            "job_type": "gap_analysis",
            "status": JOB_STATUS_PENDING,
            "request": {
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
                "save_results": False,
            },
        }
    )

    assert state["status"] == JOB_STATUS_COMPLETED
    assert job_storage.update_job_status.call_args_list[0] == call("job-1", JOB_STATUS_RUNNING)
    completed_call = job_storage.update_job_status.call_args_list[-1]
    assert completed_call.args[0:2] == ("job-1", JOB_STATUS_COMPLETED)
    assert completed_call.kwargs["result"]["run_type"] == "gap_analysis_v4"

    compliance_storage.write_compliance_result.assert_awaited_once()
    result_doc = compliance_storage.write_compliance_result.await_args.args[0]
    assert result_doc["policy_document_id"] == "pol-1"
    assert result_doc["company_name"] == "ExampleCo"
    assert result_doc["run_types"] == ["gap_v4"]
    assert result_doc["run_type"] == "gap_analysis_v4"
    assert result_doc["version"] == "v4"
    assert result_doc["retrieval_metadata"]["statute_pairs_matched"] == 1

    compliance_storage.write_compliance_run_log.assert_awaited_once()
    log_doc = compliance_storage.write_compliance_run_log.await_args.args[0]
    assert log_doc["policy_document_id"] == "pol-1"
    assert log_doc["statute_chunk_ids_used"] == ["stat-1"]
    assert log_doc["run_type"] == "gap_analysis_v4"
    assert log_doc["summary"]["addressed"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gap_analysis_graph_failure_skips_partial_persistence(
    anyio_backend: str,
) -> None:
    """Failed gap jobs are marked failed without writing misleading partial results."""
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    compliance_storage.write_compliance_run_log = AsyncMock()

    async def run_gap_analysis(_request):
        raise RuntimeError("vector search unavailable")

    with patch.dict("sys.modules", {"langgraph.graph": None}):
        graph = build_gap_analysis_graph(run_gap_analysis, job_storage, compliance_storage)

    state = await graph.ainvoke(
        {
            "job_id": "job-2",
            "job_type": "gap_analysis",
            "status": JOB_STATUS_PENDING,
            "request": {
                "policy_document_id": "pol-1",
                "applicable_jurisdictions": ["CA"],
            },
        }
    )

    assert state["status"] == JOB_STATUS_FAILED
    assert state["error"] == "vector search unavailable"
    assert job_storage.update_job_status.call_args_list[0] == call("job-2", JOB_STATUS_RUNNING)
    job_storage.update_job_status.assert_called_with(
        "job-2",
        JOB_STATUS_FAILED,
        error="vector search unavailable",
    )
    compliance_storage.write_compliance_result.assert_not_awaited()
    compliance_storage.write_compliance_run_log.assert_not_awaited()


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
