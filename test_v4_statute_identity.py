"""Regression tests for v4 gap identity: displayed statute ids, dedupe, and jurisdiction."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


def _config():
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.statute_database = ""
    config.policies_collection = "policies"
    config.policy_document_id_field = "document_id"
    config.statute_sub_topic_embeddings_collection = "statute_sub_topic_embeddings"
    config.policy_legal_embeddings_collection = "policy_legal_embeddings"
    config.category_mapping_collection = "category_mapping"
    config.adaptive_feedback_enabled = False
    config.default_jurisdictions = ["CA"]
    return config


def _llm_result() -> dict[str, Any]:
    return {
        "status": "missing",
        "policy_quote": None,
        "statute_quote": "",
        "requirement_summary": "Must disclose",
        "conflict_description": None,
        "confidence": "low",
        "_analysis_failed": False,
    }


def _run(
    statute_docs_for_filter,
    mappings: list[dict[str, Any]],
    jurisdictions: list[str],
):
    config = _config()
    statute_filters: list[dict[str, Any]] = []
    context_ids: list[Any] = []

    def statute_find(query, projection=None):
        if "document_id" in query and query.get("category") != "consumer_rights":
            context_ids.append(query["document_id"])
            return []
        if query.get("category") == "consumer_rights":
            statute_filters.append(dict(query))
            return list(statute_docs_for_filter(query))
        raise AssertionError(f"unexpected statute query: {query}")

    statute_coll = MagicMock()
    statute_coll.find.side_effect = statute_find

    policies = MagicMock()
    policies.find_one.return_value = {
        "document_id": "pol-1",
        "text": "We disclose personal information.",
        "company_name": "Acme",
    }

    legal = MagicMock()
    legal.count_documents.return_value = 1
    legal.find.return_value = [{"chunk_text": "We disclose personal information."}]

    cat_map = MagicMock()
    cat_map.find.return_value = mappings

    results = MagicMock()
    colls = {
        "policies": policies,
        "statute_sub_topic_embeddings": statute_coll,
        "policy_legal_embeddings": legal,
        "category_mapping": cat_map,
        "compliance_results": results,
        "compliance_run_log": MagicMock(),
    }
    db = MagicMock()
    db.__getitem__.side_effect = colls.__getitem__
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(return_value=_llm_result())

    service = GapAnalysisServiceV4(mongo, config, llm, limiter, critic=None)
    request = GapAnalysisRequest(
        policy_document_id="pol-1",
        applicable_jurisdictions=jurisdictions,
        save_results=False,
    )
    response = asyncio.run(service.run(request))
    return response, llm, statute_filters, context_ids, results


def test_v4_statute_reference_stringifies_missing_ids() -> None:
    """A blank header stores str(resolved id), so a missing chain becomes the label 'None'."""

    def docs_for(_query):
        return [
            {"_id": "real-id", "subtopic_text": "body"},
            {"_id": "zero-id", "document_id": 0, "subtopic_text": "body"},
            {
                "_id": "src-id",
                "document_id": False,
                "source_id": "src-9",
                "subtopic_text": "body",
            },
            {"_id": "hdr-id", "header_text": "§ 1798.100", "subtopic_text": "body"},
            {"subtopic_text": "body"},
            {"subtopic_text": "again"},
        ]

    response, llm, _filters, context_ids, results = _run(
        docs_for,
        [{"statute_category": "consumer_rights", "policy_categories": ["notice"]}],
        ["CA"],
    )

    assert [gap.statute_reference for gap in response.gaps] == [
        "None",
        "None",
        "src-9",
        "§ 1798.100",
        "None",
    ]
    assert [gap.statute_chunk_id for gap in response.gaps] == [
        "real-id",
        "zero-id",
        "src-id",
        "hdr-id",
        None,
    ]
    assert response.statute_chunk_ids_used == ["real-id", "zero-id", "src-id", "hdr-id", ""]
    assert response.retrieval_metadata.statute_items_considered == 6
    assert response.retrieval_metadata.statute_pairs_matched == 5
    assert llm.gap_check_v4.await_count == 5
    assert context_ids == ["src-9"]
    results.insert_one.assert_not_called()


def test_v4_duplicate_statute_id_counts_but_skips_second_llm_call() -> None:
    """The same mapping and _id is counted twice and sent to the model once."""

    def docs_for(_query):
        return [
            {
                "_id": "dup-1",
                "header_text": "First header",
                "subtopic_text": "First body",
                "jurisdiction": "CA",
            },
            {
                "_id": "dup-1",
                "header_text": "Second header",
                "subtopic_text": "Second body",
                "jurisdiction": "CA",
            },
        ]

    response, llm, _filters, context_ids, _results = _run(
        docs_for,
        [{"statute_category": "consumer_rights", "policy_categories": ["notice"]}],
        ["CA"],
    )

    assert len(response.gaps) == 1
    assert response.gaps[0].statute_reference == "First header"
    assert response.gaps[0].statute_chunk_id == "dup-1"
    assert response.retrieval_metadata.statute_items_considered == 2
    assert response.retrieval_metadata.statute_pairs_matched == 1
    assert response.statute_chunk_ids_used == ["dup-1"]
    assert llm.gap_check_v4.await_count == 1
    sent = llm.gap_check_v4.await_args.kwargs["statutory_requirement"]
    assert "First body" in sent
    assert "Second body" not in sent
    assert context_ids == []


def test_v4_jurisdiction_keeps_raw_values() -> None:
    """Missing and numeric 0 use the first requested jurisdiction; other values are stringified."""

    def docs_for(_query):
        return [
            {"_id": "missing", "header_text": "Missing"},
            {"_id": "zero", "header_text": "Zero", "jurisdiction": 0},
            {"_id": "false", "header_text": "False", "jurisdiction": False},
            {"_id": "str-zero", "header_text": "Str zero", "jurisdiction": "0"},
            {"_id": "twelve", "header_text": "Twelve", "jurisdiction": 12},
            {"_id": "padded", "header_text": "Padded", "jurisdiction": "  ny "},
        ]

    response, llm, _filters, _context_ids, _results = _run(
        docs_for,
        [{"statute_category": "consumer_rights", "policy_categories": ["notice"]}],
        ["California"],
    )

    assert [gap.jurisdiction for gap in response.gaps] == [
        "California",
        "California",
        "California",
        "0",
        "12",
        "  ny ",
    ]
    assert response.applicable_jurisdictions == ["California"]
    assert llm.gap_check_v4.await_count == 6


def test_v4_mapping_subtopic_strips_the_filter_and_keeps_the_dedupe_key() -> None:
    """Whitespace sub_topic is not a Mongo filter, but the raw value still splits dedupe keys."""

    def docs_for(query):
        if query.get("sub_topic") == "sale":
            return [
                {
                    "_id": "sale-only",
                    "header_text": "Sale header",
                    "subtopic_text": "sale rule",
                    "jurisdiction": "CA",
                }
            ]
        return [
            {
                "_id": "shared",
                "header_text": "Shared header",
                "subtopic_text": "shared rule",
                "jurisdiction": "CA",
            }
        ]

    response, llm, filters, _context_ids, _results = _run(
        docs_for,
        [
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": "   ",
            },
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": None,
            },
            {
                "statute_category": "consumer_rights",
                "policy_categories": ["notice"],
                "sub_topic": "  sale  ",
            },
        ],
        ["CA"],
    )

    assert filters == [
        {"category": "consumer_rights"},
        {"category": "consumer_rights"},
        {"category": "consumer_rights", "sub_topic": "sale"},
    ]
    assert [gap.statute_reference for gap in response.gaps] == [
        "Shared header",
        "Shared header",
        "Sale header",
    ]
    assert response.retrieval_metadata.statute_items_considered == 3
    assert response.retrieval_metadata.statute_pairs_matched == 3
    assert llm.gap_check_v4.await_count == 3
