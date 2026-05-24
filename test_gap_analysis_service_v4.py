"""Regression tests for category-mapping-driven V4 gap analysis."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(
    *,
    policy_doc: dict,
    policy_legal_docs: list[dict],
    statute_docs: list[dict],
    mappings: list[dict] | None = None,
    alternate_mappings: list[dict] | None = None,
):
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.statute_database = "privacy-compliance"
    config.adaptive_feedback_enabled = False
    config.category_mapping_collection = "category_mapping"

    policies_coll = MagicMock(name="policies")
    policies_coll.find_one.return_value = policy_doc

    policy_legal_coll = MagicMock(name="policy_legal_embeddings")
    policy_legal_coll.count_documents.return_value = len(policy_legal_docs)
    policy_legal_coll.find.return_value = policy_legal_docs

    statute_coll = MagicMock(name="statute_sub_topic_embeddings")

    def _find_statutes(query, projection=None):
        category = query.get("category")
        if isinstance(category, dict) and "$in" in category:
            return []
        return statute_docs

    statute_coll.find.side_effect = _find_statutes

    category_mapping_coll = MagicMock(name="category_mapping")
    category_mapping_coll.find.return_value = list(mappings or [])

    alternate_mapping_coll = MagicMock(name="category_mappings")
    alternate_mapping_coll.find.return_value = list(alternate_mappings or [])

    collections = {
        config.policies_collection: policies_coll,
        config.policy_legal_embeddings_collection: policy_legal_coll,
        config.statute_sub_topic_embeddings_collection: statute_coll,
        "category_mapping": category_mapping_coll,
        "category_mappings": alternate_mapping_coll,
        config.compliance_results_collection: MagicMock(name="compliance_results"),
        config.compliance_run_log_collection: MagicMock(name="compliance_run_log"),
    }

    database = MagicMock(name="privacy-compliance-db")
    database.__getitem__.side_effect = lambda name: collections[name]
    mongo = MagicMock(name="mongo")
    mongo.__getitem__.return_value = database

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=rate_limiter,
    )
    return service, llm, category_mapping_coll, alternate_mapping_coll


def test_v4_downgrades_unbound_addressed_quote_and_respects_num_rows() -> None:
    """Unsupported addressed quotes should not pass as covered requirements."""
    service, llm, _, _ = _build_service(
        policy_doc={
            "document_id": "policy-1",
            "text": "We explain how consumers may submit access and deletion requests.",
            "company_name": "ExampleCo",
        },
        policy_legal_docs=[
            {
                "document_id": "policy-1",
                "category": "consumer_requests",
                "chunk_text": "Consumers may submit access and deletion requests.",
            }
        ],
        statute_docs=[
            {
                "_id": "stat-1",
                "document_id": "ccpa",
                "category": "consumer_rights",
                "sub_topic": "access",
                "header_text": "CCPA § 1798.100",
                "subtopic_text": "Businesses must disclose access rights.",
                "requirement_summary": "Access disclosures are required.",
                "jurisdiction": "CA",
            },
            {
                "_id": "stat-2",
                "document_id": "ccpa",
                "category": "consumer_rights",
                "sub_topic": "access",
                "header_text": "CCPA § 1798.105",
                "subtopic_text": "Businesses must disclose deletion rights.",
                "requirement_summary": "Deletion disclosures are required.",
                "jurisdiction": "CA",
            },
        ],
        mappings=[
            {
                "statute_category": "consumer_rights",
                "sub_topic": "access",
                "policy_categories": ["consumer_requests"],
            }
        ],
    )
    llm.gap_check_v4.return_value = {
        "status": "addressed",
        "policy_quote": "We sell all personal information forever.",
        "requirement_summary": "Access disclosures are required.",
        "statute_quote": "Businesses must disclose access rights.",
        "confidence": "high",
    }

    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
                num_rows=1,
            )
        )
    )

    assert response.company_name == "ExampleCo"
    assert response.retrieval_metadata.statute_items_considered == 1
    assert response.summary.total_requirements == 1
    assert response.summary.missing == 1
    assert response.summary.addressed == 0
    assert llm.gap_check_v4.await_count == 1

    gap = response.gaps[0]
    assert gap.status == "missing"
    assert gap.policy_quote is None
    assert gap.citation_binding_failed is True
    assert gap.conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )


def test_v4_uses_plural_category_mapping_fallback() -> None:
    """The plural collection fallback keeps V4 analysis alive during migration."""
    service, llm, singular_mapping, plural_mapping = _build_service(
        policy_doc={
            "document_id": "policy-1",
            "text": "We provide deletion rights.",
        },
        policy_legal_docs=[
            {
                "document_id": "policy-1",
                "category": "consumer_requests",
                "chunk_text": "We provide deletion rights.",
            }
        ],
        statute_docs=[
            {
                "_id": "stat-1",
                "document_id": "ccpa",
                "category": "consumer_rights",
                "sub_topic": "deletion",
                "header_text": "CCPA § 1798.105",
                "subtopic_text": "Businesses must disclose deletion rights.",
                "requirement_summary": "Deletion disclosures are required.",
                "jurisdiction": "CA",
            }
        ],
        mappings=[],
        alternate_mappings=[
            {
                "statute_category": "consumer_rights",
                "sub_topic": "deletion",
                "policy_categories": ["consumer_requests"],
            }
        ],
    )
    llm.gap_check_v4.return_value = {
        "status": "addressed",
        "policy_quote": "We provide deletion rights.",
        "requirement_summary": "Deletion disclosures are required.",
        "statute_quote": "Businesses must disclose deletion rights.",
        "confidence": "high",
    }

    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="policy-1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    )

    singular_mapping.find.assert_called_once()
    plural_mapping.find.assert_called_once()
    assert response.summary.total_requirements == 1
    assert response.summary.addressed == 1
    assert response.gaps[0].citation_binding_failed is None
