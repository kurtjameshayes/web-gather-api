"""Focused regression tests for the v4 gap-analysis service."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(
    *,
    policy_text: str,
    policy_chunk: str,
    llm_result: dict,
    critic: MagicMock | None = None,
) -> tuple[GapAnalysisServiceV4, MagicMock, dict[str, MagicMock], MagicMock]:
    config = load_config()
    config.adaptive_feedback_enabled = True

    collections: dict[str, MagicMock] = {}
    db = MagicMock()

    def _collection(name: str) -> MagicMock:
        return collections.setdefault(name, MagicMock(name=name))

    db.__getitem__.side_effect = _collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    policies = _collection(config.policies_collection)
    policies.find_one.return_value = {
        config.policy_document_id_field: "pol-1",
        "company_name": "Acme",
        "text": policy_text,
    }

    policy_legal = _collection(config.policy_legal_embeddings_collection)
    policy_legal.count_documents.return_value = 1
    policy_legal.find.return_value = [{"chunk_text": policy_chunk}]

    category_mapping = _collection(config.category_mapping_collection)
    category_mapping.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["consumer_rights"],
            "sub_topic": "right_to_delete",
        }
    ]

    statute = _collection(config.statute_sub_topic_embeddings_collection)
    statute_doc = {
        "_id": "stat-1",
        "category": "consumer_rights",
        "sub_topic": "right_to_delete",
        "document_id": "ccpa",
        "jurisdiction": "CA",
        "header_text": "CCPA Section 1798.105",
        "subtopic_text": "A consumer has the right to request deletion.",
        "requirement_summary": "Right to delete",
    }
    context_doc = {
        "_id": "ctx-1",
        "category": "definitions",
        "document_id": "ccpa",
        "header_text": "Definitions",
        "subtopic_text": "Consumer means a California resident.",
    }

    def _find_statute_docs(query: dict, projection: dict | None = None) -> list[dict]:
        category = query.get("category")
        if category == "consumer_rights":
            return [statute_doc]
        if isinstance(category, dict) and category.get("$in"):
            return [context_doc]
        return []

    statute.find.side_effect = _find_statute_docs

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=llm_result)

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=rate_limiter,
        critic=critic,
    )
    return service, config, collections, llm


def test_run_downgrades_addressed_gap_when_policy_quote_is_not_grounded() -> None:
    service, _config, _collections, llm = _build_service(
        policy_text="We honor verified deletion requests within 45 days.",
        policy_chunk="We honor verified deletion requests within 45 days.",
        llm_result={
            "status": "addressed",
            "policy_quote": "We sell personal information freely.",
            "statute_quote": "A consumer has the right to request deletion.",
            "requirement_summary": "Right to delete",
            "confidence": "high",
        },
    )

    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    )

    assert len(response.gaps) == 1
    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert response.summary.missing == 1
    assert response.summary.addressed == 0

    llm.gap_check_v4.assert_awaited_once()


def test_run_injects_active_feedback_and_records_usage_after_persistence() -> None:
    critic = MagicMock()
    feedback_docs = [
        {
            "_id": "fb-1",
            "suggestions": [
                {"instruction": "Only accept verbatim policy quotes."},
            ],
        }
    ]
    critic.get_active_feedback.return_value = feedback_docs
    critic.format_feedback_for_prompt.return_value = "Prior feedback text"
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    service, config, collections, llm = _build_service(
        policy_text="We honor verified deletion requests within 45 days.",
        policy_chunk="We honor verified deletion requests within 45 days.",
        llm_result={
            "status": "addressed",
            "policy_quote": "We honor verified deletion requests within 45 days.",
            "statute_quote": "A consumer has the right to request deletion.",
            "requirement_summary": "Right to delete",
            "confidence": "high",
        },
        critic=critic,
    )

    with patch("uuid.uuid4", return_value="run-123"):
        response = asyncio.run(
            service.run(
                GapAnalysisRequest(
                    policy_document_id="pol-1",
                    applicable_jurisdictions=["CA"],
                    save_results=True,
                )
            )
        )

    critic.get_active_feedback.assert_called_once_with("pol-1")
    critic.format_feedback_for_prompt.assert_called_once_with(feedback_docs)
    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.await_args.kwargs["adaptive_feedback"] == "Prior feedback text"

    result_doc = collections[config.compliance_results_collection].insert_one.call_args[0][0]
    assert result_doc["_id"] == "run-123"
    assert result_doc["gaps"][0]["status"] == "addressed"
    collections[config.compliance_run_log_collection].insert_one.assert_called_once()

    critic.record_feedback_usage.assert_awaited_once_with(
        run_id="run-123",
        feedback_ids=["fb-1"],
        rendered_text="Prior feedback text",
    )
    critic.evaluate.assert_awaited_once()
    evaluated_response, run_id = critic.evaluate.await_args.args
    assert evaluated_response == response
    assert run_id == "run-123"
