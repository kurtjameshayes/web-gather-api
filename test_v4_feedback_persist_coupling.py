"""V4 critic coupling to result persistence.

Async jobs force save_results off, and the critic runs only after a run id is
assigned. These tests pin that split so a failed write or a background job
cannot silently drop or double-fire adaptive feedback.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


FEEDBACK_TEXT = "PRIOR ANALYSIS FEEDBACK\n\n- Be strict"


class _Allow:
    async def allow(self) -> bool:
        return True


def _columns() -> dict[str, MagicMock]:
    names = (
        "policies",
        "policy_legal_embeddings",
        "category_mapping",
        "statute_sub_topic_embeddings",
        "compliance_results",
        "compliance_run_log",
    )
    return {name: MagicMock(name=name) for name in names}


def _mongo(columns: dict[str, MagicMock]) -> MagicMock:
    database = MagicMock()
    database.__getitem__.side_effect = lambda name: columns[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = database
    return mongo


def _service(columns, critic):
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.compliance_database = "privacy-compliance"
    config.statute_database = ""
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We retain data",
            "statute_quote": "Retain records.",
            "requirement_summary": "Retention",
            "conflict_description": None,
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    columns["policies"].find_one.return_value = {
        "document_id": "pol-1",
        "text": "We retain data for the service.",
        "company_name": "Acme",
    }
    columns["policy_legal_embeddings"].count_documents.return_value = 1
    columns["policy_legal_embeddings"].find.return_value = [
        {"chunk_text": "indexed chunk about retention"},
    ]
    columns["category_mapping"].find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["data_retention"],
        }
    ]
    columns["statute_sub_topic_embeddings"].find.return_value = [
        {
            "_id": "stat-1",
            "document_id": "statute-9",
            "category": "consumer_rights",
            "header_text": "§ 100",
            "subtopic_text": "A business shall retain records.",
            "jurisdiction": "CA",
        }
    ]
    critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": [{"instruction": "Be strict"}]}
    ]
    critic.format_feedback_for_prompt.return_value = FEEDBACK_TEXT
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="new-fb")
    service = GapAnalysisServiceV4(_mongo(columns), config, llm, _Allow(), critic)
    return service, llm


def _request(**overrides) -> GapAnalysisRequest:
    payload = {
        "policy_document_id": "pol-1",
        "applicable_jurisdictions": ["CA"],
        "run_async": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest.model_validate(payload)


def test_save_results_false_injects_feedback_and_skips_critic() -> None:
    """Background runs keep the prompt injection and skip the critic writes."""
    columns = _columns()
    critic = MagicMock()
    service, llm = _service(columns, critic)

    response = asyncio.run(service.run(_request(save_results=False)))

    assert response.summary.addressed == 1
    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == FEEDBACK_TEXT
    critic.get_active_feedback.assert_called_once_with("pol-1")
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_not_awaited()
    columns["compliance_results"].insert_one.assert_not_called()
    columns["compliance_run_log"].insert_one.assert_not_called()


def test_result_insert_failure_still_records_feedback() -> None:
    """run_id is assigned before insert, so a failed result write still critiques."""
    columns = _columns()
    critic = MagicMock()
    service, _llm = _service(columns, critic)
    captured: list[str] = []

    def fail_insert(doc):
        captured.append(doc["_id"])
        raise RuntimeError("write failed")

    columns["compliance_results"].insert_one.side_effect = fail_insert

    response = asyncio.run(service.run(_request(save_results=True)))

    assert response.summary.addressed == 1
    assert response.policy_document_id == "pol-1"
    assert captured
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=captured[0],
        feedback_ids=["fb-1"],
        rendered_text=FEEDBACK_TEXT,
    )
    critic.evaluate.assert_awaited_once()
    assert critic.evaluate.await_args.args[1] == captured[0]
    columns["compliance_run_log"].insert_one.assert_not_called()


def test_usage_log_failure_skips_evaluate_and_returns_result() -> None:
    """A feedback-log error shares the critic try-block, so evaluate does not run."""
    columns = _columns()
    critic = MagicMock()
    service, _llm = _service(columns, critic)
    critic.record_feedback_usage.side_effect = RuntimeError("log down")

    response = asyncio.run(service.run(_request(save_results=True)))

    assert response.summary.addressed == 1
    columns["compliance_results"].insert_one.assert_called_once()
    columns["compliance_run_log"].insert_one.assert_called_once()
    critic.evaluate.assert_not_awaited()


def test_async_v4_job_forces_save_results_false() -> None:
    """The v4 job starter copies the request and turns persistence off."""
    from compliance_routes_v4 import _start_v4_gap_analysis_job

    original = {"policy_document_id": "pol-1", "save_results": True, "run_async": True}
    suite = MagicMock()
    with patch("compliance_routes_v4.start_gap_analysis_job", return_value="job-1") as start, patch(
        "compliance_routes_v4._get_gap_analysis_v4_service", return_value=MagicMock()
    ), patch("compliance_routes_v4._get_job_storage", return_value=MagicMock()), patch(
        "compliance_routes_v4._get_suite_service", return_value=suite
    ):
        job_id = _start_v4_gap_analysis_job(original)

    assert job_id == "job-1"
    passed = start.call_args.args[0]
    assert passed["save_results"] is False
    assert passed["policy_document_id"] == "pol-1"
    assert original["save_results"] is True
    assert start.call_args.kwargs["compliance_storage"] is suite._storage
