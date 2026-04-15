"""Regression tests for adaptive-feedback behavior in GapAnalysisServiceV4."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Keep tests lightweight when optional SDKs are unavailable.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(*, feedback_docs: list[dict], save_results: bool = True):
    config = load_config()
    config.adaptive_feedback_enabled = True

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "Consumers may request access to their data.",
            "statute_quote": "Consumers have a right to know.",
            "requirement_summary": "Right to know",
            "confidence": "high",
        }
    )

    critic = MagicMock()
    critic.get_active_feedback.return_value = feedback_docs
    critic.format_feedback_for_prompt.return_value = "PRIOR ANALYSIS FEEDBACK:\n- Use strict citation checks."
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    policy_doc = {
        "document_id": "pol-1",
        "company_name": "Acme",
        "text": "Consumers may request access to their data.",
    }
    mapping_doc = {
        "statute_category": "consumer_rights",
        "policy_categories": ["consumer_rights"],
        "sub_topic": "access",
    }
    statute_requirement_doc = {
        "_id": "stat-1",
        "document_id": "doc-1",
        "header_text": "CCPA § 1798.100",
        "subtopic_text": "Consumers have a right to know.",
        "requirement_summary": "Right to know",
        "jurisdiction": "CA",
    }
    statute_context_doc = {
        "category": "definitions",
        "header_text": "Definitions",
        "subtopic_text": "Personal information means information that identifies a consumer.",
    }

    policies_coll = MagicMock()
    policies_coll.find_one.side_effect = lambda query: policy_doc if query.get("document_id") == "pol-1" else None

    policy_legal_coll = MagicMock()
    policy_legal_coll.count_documents.return_value = 1
    policy_legal_coll.find.return_value = [{"chunk_text": policy_doc["text"]}]

    category_mapping_coll = MagicMock()
    category_mapping_coll.find.return_value = [mapping_doc]

    statute_coll = MagicMock()

    def _statute_find(query, projection=None):
        if query.get("category") == "consumer_rights":
            return [statute_requirement_doc]
        if query.get("document_id") == "doc-1":
            return [statute_context_doc]
        return []

    statute_coll.find.side_effect = _statute_find

    results_coll = MagicMock()
    run_log_coll = MagicMock()

    db = MagicMock()
    collection_map = {
        config.policies_collection: policies_coll,
        config.policy_legal_embeddings_collection: policy_legal_coll,
        config.category_mapping_collection: category_mapping_coll,
        config.statute_sub_topic_embeddings_collection: statute_coll,
        config.compliance_results_collection: results_coll,
        config.compliance_run_log_collection: run_log_coll,
    }
    db.__getitem__.side_effect = lambda name: collection_map[name]

    mongo = MagicMock()
    mongo.__getitem__.return_value = db

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
        save_results=save_results,
        run_async=False,
    )

    return service, req, critic, llm, results_coll


@pytest.mark.anyio
async def test_run_injects_feedback_and_records_usage_and_evaluation():
    feedback_docs = [{"_id": "fb-1", "suggestions": [{"instruction": "Be strict with citations"}]}]
    service, req, critic, llm, results_coll = _build_service(feedback_docs=feedback_docs, save_results=True)

    response = await service.run(req)

    assert response.summary.total_requirements == 1
    assert response.summary.addressed == 1
    critic.get_active_feedback.assert_called_once_with("pol-1")
    critic.format_feedback_for_prompt.assert_called_once_with(feedback_docs)
    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.call_args.kwargs["adaptive_feedback"] == critic.format_feedback_for_prompt.return_value

    run_id = results_coll.insert_one.call_args.args[0]["_id"]
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=run_id,
        feedback_ids=["fb-1"],
        rendered_text=critic.format_feedback_for_prompt.return_value,
    )
    critic.evaluate.assert_awaited_once_with(response, run_id)


@pytest.mark.anyio
async def test_run_evaluates_without_usage_log_when_no_feedback_docs():
    service, req, critic, _, results_coll = _build_service(feedback_docs=[], save_results=True)

    response = await service.run(req)

    run_id = results_coll.insert_one.call_args.args[0]["_id"]
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_awaited_once_with(response, run_id)


@pytest.mark.anyio
async def test_run_skips_adaptive_feedback_post_hooks_when_results_not_saved():
    feedback_docs = [{"_id": "fb-1", "suggestions": [{"instruction": "Be strict with citations"}]}]
    service, req, critic, _, results_coll = _build_service(feedback_docs=feedback_docs, save_results=False)

    await service.run(req)

    results_coll.insert_one.assert_not_called()
    critic.record_feedback_usage.assert_not_awaited()
    critic.evaluate.assert_not_awaited()
