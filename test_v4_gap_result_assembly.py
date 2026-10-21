"""V4 gap-analysis result assembly that route tests do not reach.

Pins policy-text fallback, canned missing descriptions, quote truncation,
and the guards that run before statute retrieval.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import pytest

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4, GapAnalysisServiceV4Error

CANNED_MISSING = (
    "The policy does not contain provisions that address this statutory requirement."
)


def _mongo():
    collections: dict = {}
    client = MagicMock()

    def get_db(name):
        db = MagicMock()

        def get_coll(coll_name):
            key = (name, coll_name)
            if key not in collections:
                coll = MagicMock()
                coll.find.return_value = []
                coll.find_one.return_value = None
                coll.count_documents.return_value = 1
                collections[key] = coll
            return collections[key]

        db.__getitem__.side_effect = get_coll
        return db

    client.__getitem__.side_effect = get_db

    def coll(db_name, coll_name):
        return client[db_name][coll_name]

    return client, collections, coll


def _service(mongo, llm, *, allow=True, quote_chars=300):
    cfg = load_config()
    cfg.adaptive_feedback_enabled = False
    cfg.max_statute_quote_chars = quote_chars
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=allow)
    service = GapAnalysisServiceV4(mongo, cfg, llm, limiter, critic=None)
    return service, cfg


def _request(**overrides) -> GapAnalysisRequest:
    payload = {
        "policy_document_id": "p1",
        "applicable_jurisdictions": ["CA"],
        "save_results": False,
    }
    payload.update(overrides)
    return GapAnalysisRequest(**payload)


def test_rate_limit_runs_before_policy_lookup():
    mongo, collections, _coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock()
    service, _cfg = _service(mongo, llm, allow=False)

    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        asyncio.run(service.run(_request()))

    assert exc.value.status_code == 429
    assert collections == {}
    llm.gap_check_v4.assert_not_awaited()


def test_missing_policy_ignores_request_collection_and_stops():
    mongo, collections, coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock()
    service, cfg = _service(mongo, llm)
    policies = coll(cfg.compliance_database, cfg.policies_collection)
    policies.find_one.return_value = None

    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        asyncio.run(service.run(_request(policy_collection="other_policies")))

    assert exc.value.status_code == 404
    assert str(exc.value) == "Policy not found: p1"
    assert (cfg.compliance_database, "other_policies") not in collections
    assert policies.find_one.call_args_list[0].args == ({cfg.policy_document_id_field: "p1"},)
    llm.gap_check_v4.assert_not_awaited()


def test_unindexed_policy_is_400_before_mappings():
    mongo, collections, coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock()
    service, cfg = _service(mongo, llm)
    db_name = cfg.compliance_database
    policies = coll(db_name, cfg.policies_collection)
    policies.find_one.return_value = {
        cfg.policy_document_id_field: "p1",
        "text": "We keep data.",
    }
    legal = coll(db_name, cfg.policy_legal_embeddings_collection)
    legal.count_documents.return_value = 0

    with pytest.raises(GapAnalysisServiceV4Error) as exc:
        asyncio.run(service.run(_request()))

    assert exc.value.status_code == 400
    assert str(exc.value) == (
        f"Policy not indexed. Index into {cfg.policy_legal_embeddings_collection} first."
    )
    legal.count_documents.assert_called_once_with(
        {cfg.policy_document_id_field: {"$in": ["p1"]}}
    )
    coll(db_name, cfg.category_mapping_collection).find.assert_not_called()
    statutes = coll(db_name, cfg.statute_sub_topic_embeddings_collection)
    statutes.find.assert_not_called()
    llm.gap_check_v4.assert_not_awaited()


def test_chunk_fallback_joins_policies_and_skips_irrelevant_mappings():
    mongo, collections, coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": "Keep records. TAIL --- Second policy.",
            "statute_quote": "Duty.",
            "requirement_summary": "From the model",
            "conflict_description": "not missing",
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    service, cfg = _service(mongo, llm)
    db_name = cfg.compliance_database
    id_field = cfg.policy_document_id_field
    policies = {
        "p1": {
            id_field: "p1",
            "text": "  ",
            "company_name": 12,
            "policy_chunks": [
                {"chunk_text": "  Keep records.  "},
                "not-a-chunk",
                {"chunk_text": "TAIL"},
            ],
        },
        "p2": {
            id_field: "p2",
            "text": "Second policy.",
            "company_name": "Acme",
        },
    }
    policy_coll = coll(db_name, cfg.policies_collection)
    policy_coll.find_one.side_effect = lambda query: policies.get(query.get(id_field))

    statute_calls = []

    def statute_find(filt, proj=None):
        statute_calls.append(filt)
        if filt.get("category") == "consumer_rights":
            return [
                {
                    "_id": "s1",
                    "document_id": "stat-1",
                    "header_text": "Header",
                    "subtopic_text": "Body text",
                    "requirement_summary": "stored",
                    "jurisdiction": "CA",
                }
            ]
        return []

    statutes = coll(db_name, cfg.statute_sub_topic_embeddings_collection)
    statutes.find.side_effect = statute_find
    mappings = coll(db_name, cfg.category_mapping_collection)
    mappings.find.return_value = [
        {
            "statute_category": "definitions",
            "policy_categories": ["notice"],
            "sub_topic": "defined",
        },
        {
            "statute_category": "consumer_rights",
            "policy_categories": [],
            "sub_topic": "empty-cats",
        },
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["notice"],
            "sub_topic": "keep",
        },
    ]

    result = asyncio.run(
        service.run(_request(policy_document_id=None, policy_document_ids=["p1", "p2"]))
    )

    assert result.company_name == "Acme"
    assert result.gaps[0].status == "addressed"
    assert result.gaps[0].citation_binding_failed is None
    assert result.gaps[0].policy_quote == "Keep records. TAIL --- Second policy."
    llm.gap_check_v4.assert_awaited_once()
    assert llm.gap_check_v4.await_args.kwargs["policy_text"] == ""
    assert (
        llm.gap_check_v4.await_args.kwargs["reference_context"]
        == "(No definitions or applicability context available.)"
    )
    assert statute_calls[0] == {"category": "consumer_rights", "sub_topic": "keep"}
    assert all(call.get("sub_topic") != "empty-cats" for call in statute_calls)
    assert all(call.get("sub_topic") != "defined" for call in statute_calls)
    legal = coll(db_name, cfg.policy_legal_embeddings_collection)
    legal.count_documents.assert_called_once_with({id_field: {"$in": ["p1", "p2"]}})


def test_summary_quote_and_confidence_fallbacks():
    mongo, _collections, coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        side_effect=[
            {
                "status": "conflict",
                "policy_quote": None,
                "requirement_summary": None,
                "conflict_description": "  ",
                "confidence": "HIGH",
            },
            {
                "status": "missing",
                "policy_quote": None,
                "statute_quote": "   ",
                "requirement_summary": "  ",
                "conflict_description": "   ",
                "confidence": "low",
            },
        ]
    )
    service, cfg = _service(mongo, llm, quote_chars=10)
    db_name = cfg.compliance_database
    policies = coll(db_name, cfg.policies_collection)
    policies.find_one.return_value = {
        cfg.policy_document_id_field: "p1",
        "text": "Policy body.",
        "company_name": "Acme",
    }
    statutes = coll(db_name, cfg.statute_sub_topic_embeddings_collection)
    statutes.find.side_effect = lambda filt, proj=None: (
        [
            {
                "_id": "s1",
                "document_id": "stat-1",
                "header_text": "Header",
                "subtopic_text": "ABCDEFGHIJKLMNOP",
                "requirement_summary": "Stored duty",
                "jurisdiction": "CA",
            },
            {
                "_id": "s2",
                "document_id": "stat-2",
                "header_text": "Other",
                "subtopic_text": "ZZZZZZZZZZZZZZZZ",
                "requirement_summary": "Stored other",
                "jurisdiction": "CA",
            },
        ]
        if filt.get("category") == "consumer_rights"
        else []
    )
    mappings = coll(db_name, cfg.category_mapping_collection)
    mappings.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["notice"],
            "sub_topic": "keep",
        }
    ]

    result = asyncio.run(service.run(_request()))
    first, second = result.gaps

    assert first.requirement_summary == "Stored duty"
    assert first.statute_quote == "Header\n\nAB"
    assert first.conflict_description == "  "
    assert first.confidence is None
    assert second.requirement_summary == ""
    assert second.statute_quote is None
    assert second.conflict_description == CANNED_MISSING
    assert second.confidence == "low"
    assert result.summary.missing == 1
    assert result.summary.conflicts == 1
