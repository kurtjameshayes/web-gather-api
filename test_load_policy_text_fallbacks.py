"""ComplianceSuiteService._load_policy_text lookup and assembly contracts.

Distinct from _load_policy_from_chunks (per-chunk embeddings) and
_fetch_document_text (gap-check helper). Applicability, v1 gap analysis,
citations, and risk assessment all share this loader.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

from bson import ObjectId

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_service import ComplianceSuiteService


def _service(mongo: MagicMock) -> ComplianceSuiteService:
    config = load_config()
    return ComplianceSuiteService(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=MagicMock(),
    )


def _coll_with_find(mongo: MagicMock) -> MagicMock:
    coll = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    return coll


def test_load_policy_text_uses_document_id_field_first() -> None:
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    coll.find_one.return_value = {
        "text": "  Policy body  ",
        "company_name": "Acme",
    }
    svc = _service(mongo)
    config = load_config()

    text, company = asyncio.run(svc._load_policy_text("pol-1"))

    assert text == "Policy body"
    assert company == "Acme"
    coll.find_one.assert_called_once_with({config.policy_document_id_field: "pol-1"})


def test_load_policy_text_falls_back_to_objectid_then_string_id() -> None:
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    oid = ObjectId()
    coll.find_one.side_effect = [
        None,
        {"_id": oid, "text": "from string id"},
    ]
    svc = _service(mongo)

    text, company = asyncio.run(svc._load_policy_text("not-an-objectid"))

    assert text == "from string id"
    assert company is None
    # ObjectId() raises before the second find_one; the except path uses the raw string.
    assert coll.find_one.call_count == 2
    assert coll.find_one.call_args_list[1][0][0] == {"_id": "not-an-objectid"}


def test_load_policy_text_objectid_lookup_when_hex_id() -> None:
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    hex_id = "64b8f7b7b1f1eaf5b2f0c001"
    coll.find_one.side_effect = [
        None,
        {"text": "from oid"},
    ]
    svc = _service(mongo)

    text, _ = asyncio.run(svc._load_policy_text(hex_id))

    assert text == "from oid"
    oid_query = coll.find_one.call_args_list[1][0][0]
    assert oid_query == {"_id": ObjectId(hex_id)}


def test_load_policy_text_assembles_policy_chunks_skipping_non_dicts() -> None:
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    coll.find_one.return_value = {
        "text": "   ",
        "policy_chunks": [
            "skip-me",
            {"chunk_text": "  First  "},
            {"chunk_header_text": "ignored when chunk_text missing"},
            {"chunk_text": ""},
            {"chunk_text": "Second"},
        ],
        "company_name": 123,
    }
    svc = _service(mongo)

    text, company = asyncio.run(svc._load_policy_text("pol-1"))

    assert text == "First\n\nSecond"
    assert company is None


def test_load_policy_text_does_not_use_chunk_company_name() -> None:
    """Unlike _load_policy_from_chunks, company_name comes only from the parent doc."""
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    coll.find_one.return_value = {
        "text": "",
        "policy_chunks": [{"chunk_text": "Body", "company_name": "FromChunk"}],
    }
    svc = _service(mongo)

    text, company = asyncio.run(svc._load_policy_text("pol-1"))

    assert text == "Body"
    assert company is None


def test_load_policy_text_missing_doc_is_empty() -> None:
    mongo = MagicMock()
    coll = _coll_with_find(mongo)
    coll.find_one.return_value = None
    svc = _service(mongo)

    text, company = asyncio.run(svc._load_policy_text("missing"))

    assert text == ""
    assert company is None


def test_load_policy_text_honors_database_and_collection() -> None:
    mongo = MagicMock()
    coll = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = coll
    mongo.__getitem__.return_value = db
    coll.find_one.return_value = {"text": "custom"}
    svc = _service(mongo)

    text, _ = asyncio.run(
        svc._load_policy_text("pol-1", database="other-db", policy_collection="other_policies")
    )

    assert text == "custom"
    mongo.__getitem__.assert_called_with("other-db")
    db.__getitem__.assert_called_with("other_policies")
