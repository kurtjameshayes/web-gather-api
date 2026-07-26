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
    build_health_score_graph,
    start_gap_analysis_job,
)
from compliance_suite_schemas import HealthScoreResponse
from index_job_service import IndexJobStorage, JOB_STATUS_PENDING


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
async def test_health_score_graph_persists_result_on_success(anyio_backend: str) -> None:
    """Successful health-score jobs write compliance_results before completing."""
    response = HealthScoreResponse(
        policy_document_id="pol-1",
        company_name="Example Co",
        privacy_health_score=91,
        score_assessment="Good",
        score_breakdown={"by_jurisdiction": {"CA": 91}},
        components={"addressed": 9, "missing": 1},
        analyzed_at="2026-07-26T10:00:00+00:00",
    )
    run_health_score = AsyncMock(return_value=response)
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    graph = build_health_score_graph(run_health_score, job_storage, compliance_storage)

    result = await graph.ainvoke({
        "job_id": "job-hs-1",
        "job_type": "health_score",
        "status": JOB_STATUS_PENDING,
        "request": {
            "policy_document_id": "pol-1",
            "applicable_jurisdictions": ["CA"],
        },
    })

    result_dict = response.model_dump()
    assert result["status"] == JOB_STATUS_COMPLETED
    assert result["result"] == result_dict
    run_health_score.assert_awaited_once()
    assert run_health_score.await_args.args[0].policy_document_id == "pol-1"
    compliance_storage.write_compliance_result.assert_awaited_once()
    persisted = compliance_storage.write_compliance_result.await_args.args[0]
    assert persisted["policy_document_id"] == "pol-1"
    assert persisted["privacy_health_score"] == 91
    assert persisted["run_types"] == ["health_score"]
    assert persisted["score_assessment"] == "Good"
    job_storage.update_job_status.assert_has_calls([
        call("job-hs-1", JOB_STATUS_RUNNING),
        call("job-hs-1", JOB_STATUS_COMPLETED, result=result_dict),
    ])


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_health_score_graph_fails_without_partial_persistence(
    anyio_backend: str,
) -> None:
    """Health-score failures mark the job failed and skip compliance_results writes."""
    run_health_score = AsyncMock(side_effect=RuntimeError("score unavailable"))
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    graph = build_health_score_graph(run_health_score, job_storage, compliance_storage)

    result = await graph.ainvoke({
        "job_id": "job-hs-2",
        "job_type": "health_score",
        "status": JOB_STATUS_PENDING,
        "request": {"policy_document_id": "pol-1"},
    })

    assert result["status"] == JOB_STATUS_FAILED
    assert result["error"] == "score unavailable"
    compliance_storage.write_compliance_result.assert_not_awaited()
    job_storage.update_job_status.assert_has_calls([
        call("job-hs-2", JOB_STATUS_RUNNING),
        call("job-hs-2", JOB_STATUS_FAILED, error="score unavailable"),
    ])


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_health_score_graph_completes_when_result_write_fails(
    anyio_backend: str,
) -> None:
    """Persistence failures must not fail an otherwise successful health-score job."""
    response = HealthScoreResponse(
        policy_document_id="pol-1",
        privacy_health_score=70,
        analyzed_at="2026-07-26T10:00:00+00:00",
    )
    run_health_score = AsyncMock(return_value=response)
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock(
        side_effect=RuntimeError("mongo down")
    )
    graph = build_health_score_graph(run_health_score, job_storage, compliance_storage)

    result = await graph.ainvoke({
        "job_id": "job-hs-3",
        "job_type": "health_score",
        "status": JOB_STATUS_PENDING,
        "request": {"policy_document_id": "pol-1"},
    })

    assert result["status"] == JOB_STATUS_COMPLETED
    assert result["result"]["privacy_health_score"] == 70
    job_storage.update_job_status.assert_has_calls([
        call("job-hs-3", JOB_STATUS_RUNNING),
        call("job-hs-3", JOB_STATUS_COMPLETED, result=response.model_dump()),
    ])


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
