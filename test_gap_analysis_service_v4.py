"""Regression tests for GapAnalysisServiceV4 orchestration."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def test_v4_run_injects_adaptive_feedback_and_records_usage() -> None:
    """The adaptive loop must affect prompts and be linked to the persisted run."""
    config = load_config()
    config.adaptive_feedback_enabled = True

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

    policies_coll.find_one.return_value = {
        "document_id": "pol-1",
        "company_name": "Example Co",
        "text": "Consumers may request that we delete personal information.",
    }
    policy_legal_coll.count_documents.return_value = 1
    policy_legal_coll.find.return_value = [
        {"chunk_text": "Consumers may request that we delete personal information."}
    ]
    category_mapping_coll.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "sub_topic": "delete",
            "policy_categories": ["consumer_rights"],
        }
    ]

    def find_statutes(filter_doc: dict[str, Any], projection: Any = None) -> list[dict[str, Any]]:
        category = filter_doc.get("category")
        if category == "consumer_rights":
            return [
                {
                    "_id": "stat-1",
                    "document_id": "law-1",
                    "category": "consumer_rights",
                    "sub_topic": "delete",
                    "header_text": "Right to Delete",
                    "subtopic_text": "Consumers can request deletion of personal information.",
                    "requirement_summary": "Provide a deletion request mechanism.",
                    "jurisdiction": "CA",
                }
            ]
        if isinstance(category, dict) and category.get("$in"):
            return [
                {
                    "category": "definitions",
                    "header_text": "Definitions",
                    "subtopic_text": "Consumer means a California resident.",
                }
            ]
        return []

    statute_coll.find.side_effect = find_statutes

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "delete personal information",
            "confidence": "high",
            "requirement_summary": "Provide a deletion request mechanism.",
            "statute_quote": "Consumers can request deletion of personal information.",
        }
    )
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    critic = MagicMock()
    feedback_text = "PRIOR ANALYSIS FEEDBACK\n- Require verbatim policy support."
    critic.get_active_feedback.return_value = [{"_id": "fb-1"}]
    critic.format_feedback_for_prompt.return_value = feedback_text
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    service = GapAnalysisServiceV4(mongo, config, llm, rate_limiter, critic=critic)
    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=True,
            )
        )
    )

    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.call_args.kwargs["adaptive_feedback"] == feedback_text
    results_coll.insert_one.assert_called_once()
    run_id = results_coll.insert_one.call_args[0][0]["_id"]
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=run_id,
        feedback_ids=["fb-1"],
        rendered_text=feedback_text,
    )
    critic.evaluate.assert_awaited_once_with(response, run_id)
    assert response.policy_document_id == "pol-1"
    assert response.gaps[0].status == "addressed"
