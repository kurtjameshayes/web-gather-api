"""Regression tests for V4 num_rows truncation and multi-policy response shape."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _service(mongo, llm, rate_limiter=None):
    config = load_config()
    config.adaptive_feedback_enabled = False
    limiter = rate_limiter or MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    return GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=None,
    )


def _wire_v4_mongo(config, *, policy_docs, mappings, statute_docs, indexed_count=1):
    policies_coll = MagicMock()
    policies_coll.find_one.side_effect = policy_docs
    statute_coll = MagicMock()
    # First find() is the mapping query; later finds are CONTEXT_CATEGORIES lookups.
    statute_coll.find.side_effect = lambda *args, **kwargs: (
        list(statute_docs)
        if isinstance(args[0], dict) and args[0].get("category") == "consumer_rights"
        else []
    )
    policy_legal_coll = MagicMock()
    policy_legal_coll.count_documents.return_value = indexed_count
    policy_legal_coll.find.return_value = []
    cat_map_coll = MagicMock()
    cat_map_coll.find.return_value = mappings

    def _coll(name):
        if name == config.policies_collection:
            return policies_coll
        if name == config.statute_sub_topic_embeddings_collection:
            return statute_coll
        if name == config.policy_legal_embeddings_collection:
            return policy_legal_coll
        if name == config.category_mapping_collection:
            return cat_map_coll
        return MagicMock()

    db = MagicMock()
    db.__getitem__.side_effect = _coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db
    return mongo, statute_coll


def _missing_llm_result():
    return {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": "",
        "requirement_summary": "Requirement",
        "conflict_description": None,
        "confidence": "low",
        "_analysis_failed": False,
    }


def test_v4_num_rows_slices_statute_docs_before_llm() -> None:
    config = load_config()
    mappings = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["right_to_delete"],
            "sub_topic": "deletion",
        }
    ]
    statute_docs = [
        {"_id": "s1", "header_text": "H1", "subtopic_text": "one", "jurisdiction": "CA"},
        {"_id": "s2", "header_text": "H2", "subtopic_text": "two", "jurisdiction": "CA"},
        {"_id": "s3", "header_text": "H3", "subtopic_text": "three", "jurisdiction": "CA"},
    ]
    mongo, _ = _wire_v4_mongo(
        config,
        policy_docs=[{"document_id": "p1", "text": "policy text", "company_name": "Acme"}],
        mappings=mappings,
        statute_docs=statute_docs,
    )
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=_missing_llm_result())
    svc = _service(mongo, llm)

    req = GapAnalysisRequest(
        policy_document_id="p1",
        applicable_jurisdictions=["CA"],
        num_rows=2,
        save_results=False,
    )
    result = asyncio.run(svc.run(req))

    assert llm.gap_check_v4.await_count == 2
    assert result.retrieval_metadata.statute_items_considered == 2
    assert result.summary.total_requirements == 2
    assert result.statute_chunk_ids_used == ["s1", "s2"]
    assert result.policy_document_ids is None
    assert result.company_name == "Acme"


def test_v4_policy_document_ids_echoed_only_when_multiple() -> None:
    config = load_config()
    mappings = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["right_to_delete"],
            "sub_topic": None,
        }
    ]
    statute_docs = [
        {"_id": "s1", "header_text": "H1", "subtopic_text": "one", "jurisdiction": "CA"},
    ]
    mongo, _ = _wire_v4_mongo(
        config,
        policy_docs=[
            {"document_id": "p1", "text": "policy one"},
            {"document_id": "p2", "text": "policy two", "company_name": "Beta"},
        ],
        mappings=mappings,
        statute_docs=statute_docs,
    )
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=_missing_llm_result())
    svc = _service(mongo, llm)

    req = GapAnalysisRequest(
        policy_document_ids=["p1", "p2"],
        applicable_jurisdictions=["CA"],
        save_results=False,
    )
    result = asyncio.run(svc.run(req))

    assert result.policy_document_id == "p1"
    assert result.policy_document_ids == ["p1", "p2"]
    # First policy has no company_name; second supplies it.
    assert result.company_name == "Beta"
    assert llm.gap_check_v4.await_count == 1
