"""Regression tests for GapAnalysisServiceV4 orchestration."""
from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def test_v4_run_uses_adaptive_feedback_and_downgrades_unbound_addressed_quote() -> None:
    """Feedback must reach the LLM, and fabricated addressed quotes must not pass."""
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.adaptive_feedback_max_items = 5

    mongo = MagicMock()
    db = MagicMock()
    mongo.__getitem__.return_value = db

    policies = MagicMock()
    policy_legal_embeddings = MagicMock()
    statute_subtopics = MagicMock()
    category_mapping = MagicMock()
    compliance_results = MagicMock()
    compliance_run_log = MagicMock()
    collections: dict[str, MagicMock] = {
        config.policies_collection: policies,
        config.policy_legal_embeddings_collection: policy_legal_embeddings,
        config.statute_sub_topic_embeddings_collection: statute_subtopics,
        config.category_mapping_collection: category_mapping,
        config.compliance_results_collection: compliance_results,
        config.compliance_run_log_collection: compliance_run_log,
    }
    db.__getitem__.side_effect = lambda name: collections[name]

    policies.find_one.return_value = {
        "document_id": "pol-1",
        "company_name": "Example Co",
        "text": "Consumers may submit verified deletion requests through our privacy portal.",
    }
    policy_legal_embeddings.count_documents.return_value = 1
    policy_legal_embeddings.find.return_value = [
        {
            "chunk_text": (
                "Consumers may submit verified deletion requests through our privacy portal."
            )
        }
    ]
    category_mapping.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "sub_topic": "delete",
            "policy_categories": ["consumer_rights"],
        }
    ]

    statute_doc = {
        "_id": "stat-1",
        "document_id": "ccpa",
        "category": "consumer_rights",
        "sub_topic": "delete",
        "header_text": "Cal. Civ. Code § 1798.105",
        "subtopic_text": "A consumer may request deletion of personal information.",
        "requirement_summary": "Deletion requests",
        "jurisdiction": "CA",
    }

    def find_statutes(query: dict[str, Any], projection: dict[str, Any] | None = None):
        if query == {"category": "consumer_rights", "sub_topic": "delete"}:
            return [statute_doc]
        if query.get("category") == {"$in": ["definitions", "applicability"]}:
            return []
        return []

    statute_subtopics.find.side_effect = find_statutes

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "We delete all personal information instantly.",
            "statute_quote": "A consumer may request deletion of personal information.",
            "requirement_summary": "Deletion requests",
            "confidence": "high",
            "_analysis_failed": False,
        }
    )

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    critic = MagicMock()
    critic.get_active_feedback.return_value = [
        {
            "_id": "fb-1",
            "suggestions": [{"instruction": "Require verbatim policy quotes."}],
        }
    ]
    critic.format_feedback_for_prompt.return_value = (
        "PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy quotes."
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock(return_value="fb-new")

    service = GapAnalysisServiceV4(mongo, config, llm, rate_limiter, critic=critic)
    request = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=True,
    )

    result = asyncio.run(service.run(request))

    llm.gap_check_v4.assert_awaited_once()
    assert (
        "Require verbatim policy quotes."
        in llm.gap_check_v4.call_args.kwargs["adaptive_feedback"]
    )
    assert result.gaps[0].status == "missing"
    assert result.gaps[0].policy_quote is None
    assert result.gaps[0].citation_binding_failed is True
    assert result.summary.missing == 1
    assert result.summary.addressed == 0

    compliance_results.insert_one.assert_called_once()
    persisted_run_id = compliance_results.insert_one.call_args[0][0]["_id"]
    compliance_run_log.insert_one.assert_called_once()
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=persisted_run_id,
        feedback_ids=["fb-1"],
        rendered_text="PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy quotes.",
    )
    critic.evaluate.assert_awaited_once()
    evaluated_response, evaluated_run_id = critic.evaluate.call_args[0]
    assert evaluated_response is result
    assert evaluated_run_id == persisted_run_id
