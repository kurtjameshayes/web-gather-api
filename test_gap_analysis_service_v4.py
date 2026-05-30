"""Focused regression tests for GapAnalysisServiceV4 orchestration."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(llm_result: dict, *, policy_text: str) -> tuple[GapAnalysisServiceV4, MagicMock]:
    cfg = load_config()
    cfg.adaptive_feedback_enabled = False

    policies_coll = MagicMock()
    statute_coll = MagicMock()
    policy_legal_coll = MagicMock()
    category_mapping_coll = MagicMock()

    policies_coll.find_one.return_value = {
        "document_id": "policy-1",
        "company_name": "Acme",
        "text": policy_text,
    }
    policy_legal_coll.count_documents.return_value = 1
    policy_legal_coll.find.return_value = [{"chunk_text": policy_text}]
    category_mapping_coll.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "sub_topic": "right_to_delete",
            "policy_categories": ["consumer_rights"],
        }
    ]

    statute_doc = {
        "_id": "statute-1",
        "document_id": "ccpa",
        "category": "consumer_rights",
        "sub_topic": "right_to_delete",
        "jurisdiction": "CA",
        "header_text": "Cal. Civ. Code section 1798.105",
        "subtopic_text": "Consumers must be able to request deletion of personal information.",
        "requirement_summary": "Right to delete personal information",
    }

    def find_statute_docs(query: dict, *args: object, **kwargs: object) -> list[dict]:
        if query.get("category") == "consumer_rights":
            return [statute_doc]
        return []

    statute_coll.find.side_effect = find_statute_docs

    db_collections = {
        cfg.policies_collection: policies_coll,
        cfg.statute_sub_topic_embeddings_collection: statute_coll,
        cfg.policy_legal_embeddings_collection: policy_legal_coll,
        cfg.category_mapping_collection: category_mapping_coll,
    }
    db = MagicMock()
    db.__getitem__.side_effect = lambda name: db_collections[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=llm_result)
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV4(mongo, cfg, llm, rate_limiter)
    return service, llm


def _run_gap_analysis(service: GapAnalysisServiceV4):
    return asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
                run_async=False,
            )
        )
    )


def test_v4_downgrades_addressed_gap_when_policy_quote_is_not_grounded() -> None:
    """Hallucinated addressed citations should not survive as compliant gaps."""
    service, _ = _build_service(
        {
            "status": "addressed",
            "policy_quote": "Consumers can delete data from their dashboard instantly.",
            "confidence": "high",
            "requirement_summary": "Right to delete personal information",
            "statute_quote": "Consumers must be able to request deletion.",
        },
        policy_text=(
            "Acme collects account information. Contact privacy@example.com for privacy requests."
        ),
    )

    response = _run_gap_analysis(service)

    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert "does not contain provisions" in gap.conflict_description


def test_v4_keeps_addressed_gap_when_quote_matches_with_whitespace_normalization() -> None:
    """Verbatim quotes remain valid even when stored policy text has different spacing."""
    service, llm = _build_service(
        {
            "status": "addressed",
            "policy_quote": "Consumers may request deletion of personal information.",
            "confidence": "high",
            "requirement_summary": "Right to delete personal information",
            "statute_quote": "Consumers must be able to request deletion.",
        },
        policy_text=(
            "Acme privacy rights:\n\nConsumers   may\n"
            "request deletion of personal information."
        ),
    )

    response = _run_gap_analysis(service)

    assert response.summary.addressed == 1
    gap = response.gaps[0]
    assert gap.status == "addressed"
    assert gap.policy_quote == "Consumers may request deletion of personal information."
    assert gap.citation_binding_failed is None
    llm.gap_check_v4.assert_awaited_once()
