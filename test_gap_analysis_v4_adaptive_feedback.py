"""Regression tests for adaptive feedback wiring in GapAnalysisServiceV4."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# llm_client imports anthropic at module load; stub for deterministic unit tests.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(*, feedback_docs: list[dict], save_results: bool = True):
    """Build a minimal service + mocks for one v4 run."""
    config = load_config()
    config.default_jurisdictions = ["CA"]
    config.adaptive_feedback_enabled = True

    mock_mongo = MagicMock()
    db_mock = MagicMock()
    mock_mongo.__getitem__.return_value = db_mock

    policies_coll = MagicMock()
    statute_coll = MagicMock()
    policy_legal_coll = MagicMock()
    category_map_coll = MagicMock()
    results_coll = MagicMock()
    run_log_coll = MagicMock()

    db_mock.__getitem__.side_effect = lambda name: {
        config.policies_collection: policies_coll,
        config.statute_sub_topic_embeddings_collection: statute_coll,
        config.policy_legal_embeddings_collection: policy_legal_coll,
        config.category_mapping_collection: category_map_coll,
        config.compliance_results_collection: results_coll,
        config.compliance_run_log_collection: run_log_coll,
    }[name]

    policies_coll.find_one.return_value = {
        config.policy_document_id_field: "pol-1",
        "text": "We provide deletion rights to consumers.",
        "company_name": "Acme",
    }
    policy_legal_coll.count_documents.return_value = 1
    category_map_coll.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["rights"],
            "sub_topic": "deletion",
        }
    ]
    statute_coll.find.side_effect = [
        [
            {
                "_id": "stat-1",
                "document_id": "doc-a",
                "header_text": "CCPA § 1798.105",
                "subtopic_text": "Consumers can request deletion.",
                "requirement_summary": "Right to delete",
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
    policy_legal_coll.find.return_value = [
        {"chunk_text": "We provide deletion rights to consumers."}
    ]

    mock_llm = MagicMock()
    mock_llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "missing",
            "confidence": "high",
            "requirement_summary": "Right to delete",
        }
    )

    mock_rate_limiter = MagicMock()
    mock_rate_limiter.allow = AsyncMock(return_value=True)

    mock_critic = MagicMock()
    mock_critic.get_active_feedback.return_value = feedback_docs
    mock_critic.format_feedback_for_prompt.return_value = "PRIOR ANALYSIS FEEDBACK\n- Be stricter"
    mock_critic.record_feedback_usage = AsyncMock()
    mock_critic.evaluate = AsyncMock(return_value="feedback-new")

    service = GapAnalysisServiceV4(
        mongo_client=mock_mongo,
        config=config,
        llm_client=mock_llm,
        rate_limiter=mock_rate_limiter,
        critic=mock_critic,
    )
    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=save_results,
        run_async=False,
    )
    return service, req, mock_llm, mock_critic, results_coll


def test_v4_run_injects_feedback_and_records_usage():
    service, req, mock_llm, mock_critic, results_coll = _build_service(
        feedback_docs=[{"_id": "fb-1"}, {"_id": "fb-2"}],
        save_results=True,
    )

    asyncio.run(service.run(req))

    mock_critic.get_active_feedback.assert_called_once_with("pol-1")
    mock_critic.format_feedback_for_prompt.assert_called_once()
    assert mock_llm.gap_check_v4.await_count == 1
    assert (
        mock_llm.gap_check_v4.await_args.kwargs["adaptive_feedback"]
        == "PRIOR ANALYSIS FEEDBACK\n- Be stricter"
    )

    inserted_run_id = results_coll.insert_one.call_args[0][0]["_id"]
    mock_critic.record_feedback_usage.assert_awaited_once()
    usage_kwargs = mock_critic.record_feedback_usage.await_args.kwargs
    assert usage_kwargs["run_id"] == inserted_run_id
    assert usage_kwargs["feedback_ids"] == ["fb-1", "fb-2"]
    assert usage_kwargs["rendered_text"] == "PRIOR ANALYSIS FEEDBACK\n- Be stricter"

    mock_critic.evaluate.assert_awaited_once()
    evaluate_args = mock_critic.evaluate.await_args.args
    assert evaluate_args[1] == inserted_run_id


def test_v4_run_without_active_feedback_skips_usage_log():
    service, req, mock_llm, mock_critic, _ = _build_service(
        feedback_docs=[],
        save_results=True,
    )

    asyncio.run(service.run(req))

    mock_critic.get_active_feedback.assert_called_once_with("pol-1")
    mock_critic.format_feedback_for_prompt.assert_not_called()
    assert mock_llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == ""
    mock_critic.record_feedback_usage.assert_not_awaited()
    mock_critic.evaluate.assert_awaited_once()


def test_v4_run_without_persistence_skips_critic_post_run_hooks():
    service, req, _, mock_critic, _ = _build_service(
        feedback_docs=[{"_id": "fb-1"}],
        save_results=False,
    )

    asyncio.run(service.run(req))

    mock_critic.get_active_feedback.assert_called_once_with("pol-1")
    mock_critic.record_feedback_usage.assert_not_awaited()
    mock_critic.evaluate.assert_not_awaited()
