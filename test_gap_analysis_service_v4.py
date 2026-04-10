"""Regression tests for GapAnalysisServiceV4 high-risk paths."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _build_service(
    *,
    policy_doc: dict,
    mappings: list[dict],
    fallback_mappings: list[dict],
    statute_docs: list[dict],
    policy_chunks: list[dict],
    llm_result: dict,
) -> tuple[GapAnalysisServiceV4, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build a service instance with fully mocked db dependencies."""
    cfg = load_config()
    cfg.compliance_database = "test-db"
    cfg.statute_database = ""
    cfg.adaptive_feedback_enabled = False

    policies_coll = MagicMock()
    policies_coll.find_one.return_value = policy_doc

    statute_coll = MagicMock()

    def _statute_find(query, projection=None):  # noqa: ANN001
        if query.get("category") == "consumer_rights":
            return statute_docs
        if query.get("document_id") and isinstance(query.get("category"), dict):
            return []
        return []

    statute_coll.find.side_effect = _statute_find

    policy_legal_coll = MagicMock()
    policy_legal_coll.count_documents.return_value = 1
    policy_legal_coll.find.return_value = policy_chunks

    cat_map_coll = MagicMock()
    cat_map_coll.find.return_value = mappings

    cat_maps_plural_coll = MagicMock()
    cat_maps_plural_coll.find.return_value = fallback_mappings

    collections = {
        cfg.policies_collection: policies_coll,
        cfg.statute_sub_topic_embeddings_collection: statute_coll,
        cfg.policy_legal_embeddings_collection: policy_legal_coll,
        cfg.category_mapping_collection: cat_map_coll,
        "category_mappings": cat_maps_plural_coll,
    }
    db = MagicMock()
    db.__getitem__.side_effect = lambda name: collections[name]
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=llm_result)
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = GapAnalysisServiceV4(mongo, cfg, llm, rate_limiter, critic=None)
    return service, llm, statute_coll, cat_map_coll, cat_maps_plural_coll, policy_legal_coll


@pytest.mark.anyio
async def test_v4_downgrades_unbound_addressed_quote_to_missing() -> None:
    """Addressed results without a bound quote must be downgraded to missing."""
    service, _llm, _statute_coll, _cat_map, _cat_maps_plural, _policy_legal = _build_service(
        policy_doc={
            "document_id": "pol-1",
            "text": "We process personal information for service delivery.",
            "company_name": "Acme",
        },
        mappings=[
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": "collection",
            }
        ],
        fallback_mappings=[],
        statute_docs=[
            {
                "_id": "s-1",
                "document_id": "stat-doc-1",
                "header_text": "CCPA 1798.100",
                "subtopic_text": "The policy must disclose what categories are collected.",
                "requirement_summary": "Disclose categories collected.",
                "jurisdiction": "CA",
            }
        ],
        policy_chunks=[{"chunk_text": "We collect personal data to provide our services."}],
        llm_result={
            "status": "addressed",
            "policy_quote": "This quote does not appear in policy text",
            "confidence": "high",
            "requirement_summary": "Disclose categories collected.",
        },
    )

    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=False,
        run_async=False,
    )
    response = await service.run(req)

    assert len(response.gaps) == 1
    item = response.gaps[0]
    assert item.status == "missing"
    assert item.policy_quote is None
    assert item.citation_binding_failed is True
    assert item.conflict_description == (
        "The policy does not contain provisions that address this statutory requirement."
    )
    assert response.summary.missing == 1
    assert response.summary.addressed == 0


@pytest.mark.anyio
async def test_v4_uses_plural_category_mappings_fallback() -> None:
    """When category_mapping is empty, service should read category_mappings fallback."""
    service, llm, _statute_coll, cat_map_coll, cat_maps_plural_coll, _policy_legal = _build_service(
        policy_doc={"document_id": "pol-1", "text": "Policy text", "company_name": "Acme"},
        mappings=[],
        fallback_mappings=[
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": "collection",
            }
        ],
        statute_docs=[
            {
                "_id": "s-2",
                "document_id": "stat-doc-2",
                "header_text": "CCPA 1798.110",
                "subtopic_text": "Disclose categories and purposes.",
                "requirement_summary": "Disclose categories and purposes.",
                "jurisdiction": "CA",
            }
        ],
        policy_chunks=[{"chunk_text": "We disclose categories and purposes of processing."}],
        llm_result={"status": "missing", "policy_quote": None, "confidence": "medium"},
    )

    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=False,
        run_async=False,
    )
    response = await service.run(req)

    assert len(response.gaps) == 1
    assert response.summary.total_requirements == 1
    assert cat_map_coll.find.call_count == 1
    assert cat_maps_plural_coll.find.call_count == 1
    assert llm.gap_check_v4.await_count == 1


@pytest.mark.anyio
async def test_v4_returns_empty_response_when_no_category_mappings() -> None:
    """No mappings should return a deterministic empty response without LLM calls."""
    service, llm, statute_coll, _cat_map, _cat_maps_plural, _policy_legal = _build_service(
        policy_doc={"document_id": "pol-1", "text": "Policy text", "company_name": "Acme"},
        mappings=[],
        fallback_mappings=[],
        statute_docs=[],
        policy_chunks=[],
        llm_result={"status": "missing"},
    )

    req = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        save_results=False,
        run_async=False,
    )
    response = await service.run(req)

    assert response.gaps == []
    assert response.summary.total_requirements == 0
    assert response.summary.missing == 0
    assert response.retrieval_metadata is not None
    assert response.retrieval_metadata.statute_items_considered == 0
    assert response.retrieval_metadata.statute_pairs_matched == 0
    llm.gap_check_v4.assert_not_called()
    statute_coll.find.assert_not_called()
