"""Service-level regression tests for V4 gap analysis orchestration."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


class FakeDb:
    def __init__(self, collections: dict[str, MagicMock]) -> None:
        self._collections = collections

    def __getitem__(self, name: str) -> MagicMock:
        return self._collections[name]


class FakeMongo:
    def __init__(self, dbs: dict[str, FakeDb]) -> None:
        self._dbs = dbs

    def __getitem__(self, name: str) -> FakeDb:
        return self._dbs[name]


def _build_service() -> tuple[
    GapAnalysisServiceV4,
    MagicMock,
    AsyncMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.default_jurisdictions = ["CA"]

    policies = MagicMock()
    policies.find_one.return_value = {
        "document_id": "pol-1",
        "company_name": "Example Co",
        "text": "This policy describes account access but does not include the quoted phrase.",
    }

    statute_sub_topics = MagicMock()
    statute_sub_topics.find.side_effect = [
        [
            {
                "_id": "stat-1",
                "document_id": "ccpa",
                "category": "consumer_rights",
                "sub_topic": "access",
                "header_text": "Cal. Civ. Code Sec. 1798.100",
                "subtopic_text": "Businesses must disclose consumer access rights.",
                "requirement_summary": "Right to know",
                "jurisdiction": "CA",
            }
        ],
        [
            {
                "category": "definitions",
                "header_text": "Definitions",
                "subtopic_text": "Consumer means a California resident.",
            }
        ],
    ]

    policy_legal = MagicMock()
    policy_legal.count_documents.return_value = 1
    policy_legal.find.return_value = [
        {"chunk_text": "Consumers can access information in their account settings."}
    ]

    category_mapping = MagicMock()
    category_mapping.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "sub_topic": "access",
            "policy_categories": ["consumer_rights"],
        }
    ]

    results = MagicMock()
    run_log = MagicMock()
    db = FakeDb(
        {
            config.policies_collection: policies,
            config.statute_sub_topic_embeddings_collection: statute_sub_topics,
            config.policy_legal_embeddings_collection: policy_legal,
            config.category_mapping_collection: category_mapping,
            config.compliance_results_collection: results,
            config.compliance_run_log_collection: run_log,
        }
    )
    mongo = FakeMongo({config.compliance_database: db})

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "an exact quote that is not in the policy text",
            "confidence": "high",
            "requirement_summary": "Right to know",
            "statute_quote": "Businesses must disclose consumer access rights.",
        }
    )

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    critic = MagicMock()
    critic.get_active_feedback.return_value = [
        {
            "_id": "fb-1",
            "suggestions": [{"instruction": "Require exact policy quotes."}],
        }
    ]
    critic.format_feedback_for_prompt.return_value = (
        "PRIOR ANALYSIS FEEDBACK\n- Require exact policy quotes."
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=rate_limiter,
        critic=critic,
    )
    return service, critic, llm.gap_check_v4, results, run_log, policy_legal


def test_run_injects_feedback_downgrades_unbound_citations_and_records_usage() -> None:
    service, critic, gap_check_v4, results, run_log, policy_legal = _build_service()
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        run_async=False,
    )

    with patch("uuid.uuid4", return_value="run-123"):
        response = asyncio.run(service.run(req))

    assert response.company_name == "Example Co"
    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    assert response.gaps[0].status == "missing"
    assert response.gaps[0].policy_quote is None
    assert response.gaps[0].citation_binding_failed is True
    assert response.gaps[0].conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )

    gap_check_v4.assert_awaited_once()
    llm_kwargs = gap_check_v4.await_args.kwargs
    assert llm_kwargs["adaptive_feedback"] == (
        "PRIOR ANALYSIS FEEDBACK\n- Require exact policy quotes."
    )
    assert "Consumer means a California resident." in llm_kwargs["reference_context"]
    assert llm_kwargs["policy_text"] == (
        "Consumers can access information in their account settings."
    )
    policy_legal.find.assert_called_once_with(
        {"document_id": "pol-1", "category": {"$in": ["consumer_rights"]}},
        {"chunk_text": 1},
    )

    results.insert_one.assert_called_once()
    persisted = results.insert_one.call_args.args[0]
    assert persisted["_id"] == "run-123"
    assert persisted["summary"]["missing"] == 1
    assert persisted["gaps"][0]["citation_binding_failed"] is True
    run_log.insert_one.assert_called_once()
    assert run_log.insert_one.call_args.args[0]["statute_item_ids"] == ["stat-1"]

    critic.record_feedback_usage.assert_awaited_once_with(
        run_id="run-123",
        feedback_ids=["fb-1"],
        rendered_text="PRIOR ANALYSIS FEEDBACK\n- Require exact policy quotes.",
    )
    critic.evaluate.assert_awaited_once_with(response, "run-123")


def test_run_continues_when_adaptive_feedback_lookup_fails() -> None:
    service, critic, gap_check_v4, results, _, _ = _build_service()
    critic.get_active_feedback.side_effect = RuntimeError("temporary feedback outage")
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        run_async=False,
    )

    response = asyncio.run(service.run(req))

    assert response.summary.total_requirements == 1
    assert response.gaps[0].status == "missing"
    assert gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    results.insert_one.assert_called_once()
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_awaited_once()
