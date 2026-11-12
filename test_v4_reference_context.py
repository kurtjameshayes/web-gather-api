"""Pin v4 statute-context assembly and which text the model versus citation binding sees.

The empty-context placeholder, citation demotion, and policy-id filter shape are
covered elsewhere. This pins which statute id is used to load definitions, how
those rows are rendered, and the split between category chunks sent to the model
and the full policy used for citation binding.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import CONTEXT_CATEGORIES, GapAnalysisServiceV4

PLACEHOLDER = "(No definitions or applicability context available.)"
POLICY_QUOTE = "We keep records."
CONTEXT_PROJECTION = {
    "subtopic_text": 1,
    "requirement_summary": 1,
    "header_text": 1,
    "category": 1,
}


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

    return client, coll


def _context_rows():
    return [
        {
            "category": "definitions",
            "header_text": "  Def Title  ",
            "subtopic_text": "  ",
            "requirement_summary": "SHOULD NOT USE",
        },
        {
            "category": "definitions",
            "header_text": "",
            "subtopic_text": "",
            "requirement_summary": "  Fallback summary  ",
        },
        {
            "category": "applicability",
            "subtopic_text": "Applies to residents.",
        },
        {
            "category": "definitions",
            "header_text": "Only header",
            "subtopic_text": "",
        },
        {
            "header_text": "Missing cat",
            "subtopic_text": "Body",
        },
        {
            "category": "definitions",
            "header_text": "   ",
            "subtopic_text": "   ",
            "requirement_summary": "   ",
        },
        {
            "category": "definitions",
            "subtopic_text": 0,
            "requirement_summary": "Zero falls through",
        },
    ]


def _expected_context() -> str:
    parts = [
        "[definitions]\nDef Title",
        "Fallback summary",
        "Applies to residents.",
        "[definitions]\nOnly header",
        "[]\nMissing cat\n\nBody",
        "Zero falls through",
    ]
    return "\n\n---\n\n".join(parts)


def test_reference_context_formatting_and_binding_text_split() -> None:
    mongo, coll = _mongo()
    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock(
        return_value={
            "status": "addressed",
            "policy_quote": POLICY_QUOTE,
            "statute_quote": "Duty.",
            "requirement_summary": "From the model",
            "conflict_description": "not missing",
            "confidence": "high",
            "_analysis_failed": False,
        }
    )
    cfg = load_config()
    cfg.adaptive_feedback_enabled = False
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    service = GapAnalysisServiceV4(mongo, cfg, llm, limiter, critic=None)

    db_name = cfg.compliance_database
    id_field = cfg.policy_document_id_field
    policies = coll(db_name, cfg.policies_collection)
    policies.find_one.return_value = {
        id_field: "p1",
        "text": f"Intro. {POLICY_QUOTE} End.",
        "company_name": "Acme",
    }

    statute_calls = []

    def statute_find(filt, proj=None):
        statute_calls.append((filt, proj))
        if filt.get("category") == "consumer_rights":
            return [
                {
                    "_id": "no-doc",
                    "header_text": "No Doc",
                    "subtopic_text": "No document id",
                    "requirement_summary": "Stored",
                    "jurisdiction": "CA",
                },
                {
                    "_id": "zero-doc",
                    "document_id": 0,
                    "statute_reference": "ref-1",
                    "header_text": "Zero",
                    "subtopic_text": "Numeric zero id",
                    "requirement_summary": "Stored zero",
                    "jurisdiction": "CA",
                },
                {
                    "_id": "false-doc",
                    "document_id": False,
                    "statute_reference": 0,
                    "source_id": "src-9",
                    "header_text": "False",
                    "subtopic_text": "False id",
                    "requirement_summary": "Stored false",
                    "jurisdiction": "CA",
                },
                {
                    "_id": "padded-doc",
                    "document_id": "  doc  ",
                    "statute_reference": "ignored-ref",
                    "header_text": "Padded",
                    "subtopic_text": "Padded id",
                    "requirement_summary": "Stored padded",
                    "jurisdiction": "CA",
                },
                {
                    "_id": "string-zero",
                    "document_id": "0",
                    "statute_reference": "ignored-zero",
                    "header_text": "String zero",
                    "subtopic_text": "String zero id",
                    "requirement_summary": "Stored string zero",
                    "jurisdiction": "CA",
                },
            ]
        looked_up = filt.get("document_id")
        if looked_up == "  doc  ":
            return _context_rows()
        if looked_up in ("ref-1", "src-9", "0"):
            return []
        raise AssertionError(f"unexpected statute query: {filt}")

    statutes = coll(db_name, cfg.statute_sub_topic_embeddings_collection)
    statutes.find.side_effect = statute_find

    mappings = coll(db_name, cfg.category_mapping_collection)
    mappings.find.return_value = [
        {
            "statute_category": "consumer_rights",
            "policy_categories": ["notice", "rights"],
            "sub_topic": "access",
        }
    ]

    legal = coll(db_name, cfg.policy_legal_embeddings_collection)
    legal.find.return_value = [
        {"chunk_text": "  Keep this.  "},
        {"chunk_text": "   "},
        {"chunk_text": ""},
        {},
        {"chunk_text": "Second chunk"},
    ]

    result = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="p1",
                applicable_jurisdictions=["CA"],
                save_results=False,
            )
        )
    )

    assert llm.gap_check_v4.await_count == 5
    calls = llm.gap_check_v4.await_args_list
    assert calls[0].kwargs["reference_context"] == PLACEHOLDER
    assert calls[1].kwargs["reference_context"] == PLACEHOLDER
    assert calls[2].kwargs["reference_context"] == PLACEHOLDER
    assert calls[3].kwargs["reference_context"] == _expected_context()
    assert calls[4].kwargs["reference_context"] == PLACEHOLDER
    assert "SHOULD NOT USE" not in calls[3].kwargs["reference_context"]
    assert all(call.kwargs["adaptive_feedback"] == "" for call in calls)

    chunk_block = "Keep this.\n\n---\n\nSecond chunk"
    assert all(call.kwargs["policy_text"] == chunk_block for call in calls)
    assert POLICY_QUOTE not in chunk_block

    assert [gap.status for gap in result.gaps] == ["addressed"] * 5
    assert all(gap.policy_quote == POLICY_QUOTE for gap in result.gaps)
    assert all(gap.citation_binding_failed is None for gap in result.gaps)

    context_calls = [call for call in statute_calls if "document_id" in call[0] and call[0].get("category") != "consumer_rights"]
    assert context_calls == [
        (
            {"document_id": looked_up, "category": {"$in": list(CONTEXT_CATEGORIES)}},
            CONTEXT_PROJECTION,
        )
        for looked_up in ("ref-1", "src-9", "  doc  ", "0")
    ]

    legal.find.assert_called_with(
        {id_field: "p1", "category": {"$in": ["notice", "rights"]}},
        {"chunk_text": 1},
    )
    assert legal.find.call_count == 5
