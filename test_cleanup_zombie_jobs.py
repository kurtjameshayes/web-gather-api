"""Startup cleanup of interrupted compliance background jobs."""
from __future__ import annotations

from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_job_service import (
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    ComplianceJobStorage,
)


def test_cleanup_zombie_jobs_marks_pending_and_running_failed() -> None:
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.update_many.return_value.modified_count = 2
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    storage = ComplianceJobStorage(mock_mongo, load_config())

    count = storage.cleanup_zombie_jobs()
    assert count == 2
    mock_coll.update_many.assert_called_once()
    query, update = mock_coll.update_many.call_args[0]
    assert query == {"status": {"$in": [JOB_STATUS_PENDING, JOB_STATUS_RUNNING]}}
    assert update["$set"]["status"] == JOB_STATUS_FAILED
    assert update["$set"]["error"] == "Job interrupted by server restart."
    assert "completed_at" in update["$set"]


def test_cleanup_zombie_jobs_zero_modified_is_quiet() -> None:
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_coll.update_many.return_value.modified_count = 0
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    storage = ComplianceJobStorage(mock_mongo, load_config())

    assert storage.cleanup_zombie_jobs() == 0
