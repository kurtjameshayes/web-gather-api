"""Focused tests for GapAnalysisServiceV4 adaptive feedback paths."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service():
    """Create a v4 service with deterministic in-memory mocks."""
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.default_jurisdictions = ["CA"]

    mongo = MagicMock()
    db = MagicMock()
    mongo.__getitem__.return_value = db

    policies_coll = MagicMock()
    statute_coll = MagicMock()
    policy_legal_coll = MagicMock()
    category_mapping_coll = MagicMock()
    results_coll = MagicMock()
    run_log_coll = MagicMock()

    collections = {
        config.policies_collection: policies_coll,
        config.statute_sub_topic_embeddings_collection: statute_coll,
        config.policy_legal_embeddings_collection: policy_legal_coll,
        config.category_mapping_collection: category_mapping_coll,
        config.compliance_results_collection: results_coll,
        config.compliance_run_log_collection: run_log_coll,
    }
    db.__getitem__.side_effect = lambda name: collections[name]

    policy_id_field = config.policy_document_id_field
    policies_coll.find_one.return_value = {
        policy_id_field: "pol-1",
        "text": "Policy text for matching and citation binding.",
        "company_name": "Acme, Inc.",
    }

    policy_legal_coll.count_documents.return_value = 1
    policy_legal_coll.find.return_value = [
        {"chunk_text": "Policy text for matching and citation binding."}
    ]

    category_mapping_coll.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["consumer_rights"],
            "sub_topic": "right_to_know",
        }
    ]

    statute_doc = {
        "_id": "stat-1",
        "document_id": "doc-1",
        "header_text": "CCPA 1798.100",
        "subtopic_text": "Businesses must disclose collection categories.",
        "requirement_summary": "Right to know collected categories.",
        "jurisdiction": "CA",
        "category": "consumer_rights",
    }
    context_doc = {
        "category": "definitions",
        "header_text": "Definitions",
        "subtopic_text": "Personal information has broad statutory meaning.",
    }

    def _statute_find(query, projection=None):  # noqa: ARG001
        if "document_id" in query:
            return [context_doc]
        return [statute_doc]

    statute_coll.find.side_effect = _statute_find

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "missing",
            "confidence": "high",
        }
    )

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    critic = MagicMock()
    critic.get_active_feedback.return_value = []
    critic.format_feedback_for_prompt.return_value = "Use more exact statutory citations."
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="fb-new")

    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=rate_limiter,
        critic=critic,
    )

    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=True,
        run_async=False,
    )

    return SimpleNamespace(
        service=service,
        req=req,
        llm=llm,
        critic=critic,
        results_coll=results_coll,
        run_log_coll=run_log_coll,
    )


@pytest.mark.anyio
async def test_v4_injects_adaptive_feedback_and_records_usage():
    env = _build_service()
    env.critic.get_active_feedback.return_value = [
        {"_id": "fb-1", "suggestions": [{"instruction": "Prefer exact citations."}]}
    ]
    env.critic.format_feedback_for_prompt.return_value = "Prefer exact citations."

    with patch("uuid.uuid4", return_value="run-123"):
        response = await env.service.run(env.req)

    assert response.policy_document_id == "pol-1"
    assert len(response.gaps) == 1

    env.llm.gap_check_v4.assert_awaited_once()
    assert env.llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == "Prefer exact citations."
    env.critic.record_feedback_usage.assert_awaited_once_with(
        run_id="run-123",
        feedback_ids=["fb-1"],
        rendered_text="Prefer exact citations.",
    )
    env.critic.evaluate.assert_awaited_once()
    assert env.critic.evaluate.await_args.args[1] == "run-123"

    env.results_coll.insert_one.assert_called_once()
    persisted = env.results_coll.insert_one.call_args[0][0]
    assert persisted["_id"] == "run-123"
    env.run_log_coll.insert_one.assert_called_once()


@pytest.mark.anyio
async def test_v4_skips_usage_log_when_no_active_feedback():
    env = _build_service()
    env.critic.get_active_feedback.return_value = []

    with patch("uuid.uuid4", return_value="run-456"):
        await env.service.run(env.req)

    env.llm.gap_check_v4.assert_awaited_once()
    assert env.llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    env.critic.format_feedback_for_prompt.assert_not_called()
    env.critic.record_feedback_usage.assert_not_awaited()
    env.critic.evaluate.assert_awaited_once()
    assert env.critic.evaluate.await_args.args[1] == "run-456"


@pytest.mark.anyio
async def test_v4_tolerates_feedback_retrieval_error():
    env = _build_service()
    env.critic.get_active_feedback.side_effect = RuntimeError("temporary store failure")

    with patch("uuid.uuid4", return_value="run-789"):
        response = await env.service.run(env.req)

    assert response.summary.total_requirements == 1
    env.llm.gap_check_v4.assert_awaited_once()
    assert env.llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    env.critic.record_feedback_usage.assert_not_awaited()
    env.critic.evaluate.assert_awaited_once()
    assert env.critic.evaluate.await_args.args[1] == "run-789"
