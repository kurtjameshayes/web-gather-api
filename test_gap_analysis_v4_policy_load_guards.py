"""V4 gap-analysis policy load, index, and rate-limit regression coverage.

Open PRs for GapAnalysisServiceV4 cover citation-binding demotion, mapping
fallback, empty-mapping persist, num_rows, and adaptive-feedback. They do not
pin the pre-LLM guards: missing ids, 429, 404 lookup fallbacks, policy_legal
index check, policy_chunks assembly, or whitespace-normalized citation matches.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import (
    GapAnalysisServiceV4,
    GapAnalysisServiceV4Error,
    _citation_binding,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _req(**overrides):
    payload = {
        "policy_document_id": "pol-1",
        "save_results": False,
        "run_async": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest.model_validate(payload)


def _service(allow: bool = True, collections: dict | None = None):
    config = load_config()
    mongo = MagicMock()
    db = MagicMock()
    colls = collections or {}

    def _coll(name: str):
        if name not in colls:
            colls[name] = MagicMock()
        return colls[name]

    db.__getitem__.side_effect = _coll
    mongo.__getitem__.return_value = db
    for name in (
        config.policies_collection,
        config.statute_sub_topic_embeddings_collection,
        config.policy_legal_embeddings_collection,
        config.category_mapping_collection,
        "category_mappings",
        config.compliance_results_collection,
        config.compliance_run_log_collection,
    ):
        _coll(name)

    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=allow)
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "we obtain consent before sharing.",
            "statute_quote": "Obtain consent.",
            "requirement_summary": "Consent",
            "conflict_description": None,
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    svc = GapAnalysisServiceV4(
        mongo_client=mongo,
        config=config,
        llm_client=llm,
        rate_limiter=limiter,
        critic=None,
    )
    return svc, config, colls, llm


def test_v4_run_requires_policy_ids_even_when_schema_is_bypassed() -> None:
    svc, _, _, _ = _service()
    req = GapAnalysisRequest.model_construct(
        policy_document_id=None,
        policy_document_ids=None,
        save_results=False,
    )
    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        _run(svc.run(req))
    assert exc.value.status_code == 400
    assert "policy_document_id" in str(exc.value)


def test_v4_run_rate_limit_is_429_before_mongo_lookup() -> None:
    svc, config, colls, _ = _service(allow=False)
    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        _run(svc.run(_req()))
    assert exc.value.status_code == 429
    assert "Rate limit exceeded" in str(exc.value)
    policies = colls.get(config.policies_collection)
    if policies is not None:
        policies.find_one.assert_not_called()


def test_v4_run_policy_not_found_is_404() -> None:
    svc, config, colls, _ = _service()
    colls[config.policies_collection].find_one.return_value = None
    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        _run(svc.run(_req(policy_document_id="missing-pol")))
    assert exc.value.status_code == 404
    assert "Policy not found: missing-pol" in str(exc.value)


def test_v4_load_policy_doc_falls_back_to_objectid_then_uses_request_id_for_index_check() -> None:
    oid_hex = "507f1f77bcf86cd799439011"
    svc, config, colls, _ = _service()
    policies = colls[config.policies_collection]
    legal = colls[config.policy_legal_embeddings_collection]
    mappings = colls[config.category_mapping_collection]

    def _find_one(query):
        if query == {config.policy_document_id_field: oid_hex}:
            return None
        if query == {"_id": ObjectId(oid_hex)}:
            return {"document_id": "canonical-pol", "text": "Policy body."}
        return None

    policies.find_one.side_effect = _find_one
    legal.count_documents.return_value = 0
    mappings.find.return_value = []

    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        _run(svc.run(_req(policy_document_id=oid_hex)))
    assert exc.value.status_code == 400
    assert "Policy not indexed" in str(exc.value)
    assert config.policy_legal_embeddings_collection in str(exc.value)
    legal.count_documents.assert_called_once_with(
        {config.policy_document_id_field: {"$in": [oid_hex]}}
    )


def test_v4_load_policy_doc_falls_back_to_string__id_when_objectid_is_invalid() -> None:
    svc, config, colls, _ = _service()
    policies = colls[config.policies_collection]
    legal = colls[config.policy_legal_embeddings_collection]
    mappings = colls[config.category_mapping_collection]

    def _find_one(query):
        if query == {config.policy_document_id_field: "pol-not-oid"}:
            return None
        if query == {"_id": "pol-not-oid"}:
            return {"text": "Looked up by string _id."}
        return None

    policies.find_one.side_effect = _find_one
    legal.count_documents.return_value = 1
    mappings.find.return_value = []

    result = _run(svc.run(_req(policy_document_id="pol-not-oid")))
    assert result.gaps == []
    assert result.policy_document_id == "pol-not-oid"


def test_v4_run_not_indexed_names_policy_legal_embeddings() -> None:
    svc, config, colls, _ = _service()
    colls[config.policies_collection].find_one.return_value = {
        "document_id": "pol-1",
        "text": "A privacy policy.",
    }
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 0
    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        _run(svc.run(_req()))
    assert exc.value.status_code == 400
    assert str(exc.value) == (
        f"Policy not indexed. Index into {config.policy_legal_embeddings_collection} first."
    )


def test_v4_policy_chunks_assemble_text_for_whitespace_normalized_citation_binding() -> None:
    svc, config, colls, llm = _service()
    colls[config.policies_collection].find_one.return_value = {
        "document_id": "pol-1",
        "text": "   ",
        "company_name": "Acme Corp",
        "policy_chunks": [
            {"chunk_text": "We obtain consent before sharing."},
            "not-a-dict",
            {"chunk_text": "   "},
            {"chunk_text": "Users may delete data."},
        ],
    }
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 1
    colls[config.policy_legal_embeddings_collection].find.return_value = []
    colls[config.category_mapping_collection].find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["consent"],
            "sub_topic": "consent",
        }
    ]

    def _statute_find(query, *args, **kwargs):
        if "document_id" in query:
            return []
        return [
            {
                "_id": "stat-1",
                "document_id": "statute-doc-1",
                "header_text": "Consent",
                "subtopic_text": "Controllers must obtain consent.",
                "requirement_summary": "Obtain consent",
                "jurisdiction": "CA",
            }
        ]

    colls[config.statute_sub_topic_embeddings_collection].find.side_effect = _statute_find
    llm.gap_check_v4.return_value = {
        "status": "addressed",
        "policy_quote": "we  obtain   CONSENT before sharing.",
        "statute_quote": "Controllers must obtain consent.",
        "requirement_summary": "Consent",
        "conflict_description": None,
        "confidence": "high",
        "_analysis_failed": False,
    }

    result = _run(svc.run(_req()))
    assert result.company_name == "Acme Corp"
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.status == "addressed"
    assert gap.citation_binding_failed is None
    assert gap.policy_quote == "we  obtain   CONSENT before sharing."
    assert result.summary.addressed == 1


def test_v4_empty_binding_text_skips_citation_check() -> None:
    """When assembled policy text and legal chunks are both empty, addressed quotes are not demoted."""
    svc, config, colls, llm = _service()
    colls[config.policies_collection].find_one.return_value = {
        "document_id": "pol-1",
        "text": "",
    }
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 1
    colls[config.policy_legal_embeddings_collection].find.return_value = []
    colls[config.category_mapping_collection].find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["consent"],
            "sub_topic": "consent",
        }
    ]

    def _statute_find(query, *args, **kwargs):
        if "document_id" in query:
            return []
        return [
            {
                "_id": "stat-1",
                "header_text": "Consent",
                "subtopic_text": "Obtain consent.",
                "requirement_summary": "Obtain consent",
                "jurisdiction": "CA",
            }
        ]

    colls[config.statute_sub_topic_embeddings_collection].find.side_effect = _statute_find
    llm.gap_check_v4.return_value = {
        "status": "addressed",
        "policy_quote": "this quote is not in any policy text",
        "statute_quote": "Obtain consent.",
        "requirement_summary": "Consent",
        "conflict_description": None,
        "confidence": "high",
        "_analysis_failed": False,
    }

    result = _run(svc.run(_req()))
    assert result.gaps[0].status == "addressed"
    assert result.gaps[0].citation_binding_failed is None
    assert result.gaps[0].policy_quote == "this quote is not in any policy text"


def test_v4_whitespace_database_falls_back_to_compliance_database() -> None:
    svc, config, colls, _ = _service()
    colls[config.policies_collection].find_one.return_value = {
        "document_id": "pol-1",
        "text": "Policy.",
    }
    colls[config.policy_legal_embeddings_collection].count_documents.return_value = 1
    colls[config.category_mapping_collection].find.return_value = []

    result = _run(svc.run(_req(database="   ")))
    assert result.gaps == []
    svc.mongo.__getitem__.assert_called_with(config.compliance_database)


@pytest.mark.parametrize(
    ("quote", "text", "expected"),
    [
        ("", "policy text", False),
        ("quote", "", False),
        (None, "policy text", False),
        ("We obtain consent.", "we  OBTAIN   consent.", True),
        ("missing phrase", "unrelated policy", False),
    ],
)
def test_v4_citation_binding_normalizes_whitespace_and_case(
    quote: str | None, text: str, expected: bool
) -> None:
    assert _citation_binding(quote, text) is expected
