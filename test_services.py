"""Unit tests for compliance_job_service, index_job_service, embedder."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from compliance_config import load_config
from compliance_job_service import (
    ComplianceJobStorage,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    start_gap_analysis_job,
)
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


def test_compliance_job_storage_cleanup_zombie_jobs_marks_pending_and_running_failed() -> None:
    """Startup cleanup must fail only unfinished jobs and report the modified count."""
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.update_many.return_value.modified_count = 2
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    config = load_config()
    storage = ComplianceJobStorage(mock_mongo, config)

    with patch("compliance_job_service._iso", return_value="2026-06-30T10:00:00+00:00"):
        cleaned = storage.cleanup_zombie_jobs()

    assert cleaned == 2
    mock_coll.update_many.assert_called_once_with(
        {"status": {"$in": [JOB_STATUS_PENDING, JOB_STATUS_RUNNING]}},
        {
            "$set": {
                "status": JOB_STATUS_FAILED,
                "error": "Job interrupted by server restart.",
                "completed_at": "2026-06-30T10:00:00+00:00",
            }
        },
    )


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
