"""Regression tests for v3/v4 async gap-analysis job starter wiring."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Keep imports lightweight/deterministic without the Anthropic SDK.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_routes_v3 import _start_v3_gap_analysis_job
from compliance_routes_v4 import _start_v4_gap_analysis_job


def test_start_v3_gap_analysis_job_forces_save_results_false() -> None:
    """Async v3 jobs must disable in-service persistence to avoid duplicate writes."""
    mock_svc = MagicMock()
    mock_job_storage = MagicMock()
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock(name="compliance_storage")
    original = {
        "policy_document_id": "pol-1",
        "applicable_jurisdictions": ["CA"],
        "save_results": True,
    }

    with patch("compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_svc), patch(
        "compliance_routes_v3._get_job_storage", return_value=mock_job_storage
    ), patch("compliance_routes_v3._get_suite_service", return_value=mock_suite), patch(
        "compliance_routes_v3.start_gap_analysis_job", return_value="job-v3"
    ) as start_job:
        job_id = _start_v3_gap_analysis_job(original)

    assert job_id == "job-v3"
    # Caller payload must not be mutated in place.
    assert original["save_results"] is True

    request_dict, job_storage, runner = start_job.call_args.args
    assert request_dict["save_results"] is False
    assert request_dict["policy_document_id"] == "pol-1"
    assert job_storage is mock_job_storage
    assert start_job.call_args.kwargs["compliance_storage"] is mock_suite._storage
    assert callable(runner)


def test_start_v4_gap_analysis_job_forces_save_results_false() -> None:
    """Async v4 jobs must disable in-service persistence to avoid duplicate writes."""
    mock_svc = MagicMock()
    mock_job_storage = MagicMock()
    mock_suite = MagicMock()
    mock_suite._storage = MagicMock(name="compliance_storage")
    original = {
        "policy_document_id": "pol-2",
        "applicable_jurisdictions": ["CA", "EU"],
        "save_results": True,
    }

    with patch("compliance_routes_v4._get_gap_analysis_v4_service", return_value=mock_svc), patch(
        "compliance_routes_v4._get_job_storage", return_value=mock_job_storage
    ), patch("compliance_routes_v4._get_suite_service", return_value=mock_suite), patch(
        "compliance_routes_v4.start_gap_analysis_job", return_value="job-v4"
    ) as start_job:
        job_id = _start_v4_gap_analysis_job(original)

    assert job_id == "job-v4"
    assert original["save_results"] is True

    request_dict, job_storage, runner = start_job.call_args.args
    assert request_dict["save_results"] is False
    assert request_dict["policy_document_id"] == "pol-2"
    assert request_dict["applicable_jurisdictions"] == ["CA", "EU"]
    assert job_storage is mock_job_storage
    assert start_job.call_args.kwargs["compliance_storage"] is mock_suite._storage
    assert callable(runner)
