"""Policy resolution and statute bucketing used by the compliance suite.

These helpers decide which policy text and which statute rows citations,
risk assessment, multi-jurisdictional comparison, and the consumer-rights
router actually see. Route tests mock the suite service, so the lookup and
bucketing rules are otherwise unenforced.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

# Stub optional imports before the suite service pulls in the LLM and embedder.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from bson import ObjectId

from compliance_suite_service import ComplianceSuiteService


def _service(statute_database: str = "statutes-db") -> tuple[ComplianceSuiteService, MagicMock]:
    config = MagicMock()
    config.llm_concurrency = 1
    config.compliance_database = "privacy-compliance"
    config.statute_database = statute_database
    config.policies_collection = "policies"
    config.policy_document_id_field = "document_id"
    config.statute_sub_topic_embeddings_collection = "statute_sub_topic_embeddings"
    config.statute_jurisdiction_field = "jurisdiction"
    mongo = MagicMock()
    service = ComplianceSuiteService(
        mongo,
        config,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    return service, mongo


def _bind(mongo: MagicMock, db_name: str, coll_name: str, coll: MagicMock) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []

    def get_coll(name: str) -> MagicMock:
        seen.append((db_name, name))
        assert name == coll_name
        return coll

    db = MagicMock()
    db.__getitem__.side_effect = get_coll

    def get_db(name: str) -> MagicMock:
        assert name == db_name
        return db

    mongo.__getitem__.side_effect = get_db
    return seen


def test_load_policy_text_prefers_document_id_then_object_id() -> None:
    service, mongo = _service()
    coll = MagicMock()
    oid = ObjectId()
    _bind(mongo, "other-db", "custom_policies", coll)

    def find_one(query: dict) -> dict | None:
        if query == {"document_id": str(oid)}:
            return None
        if query == {"_id": oid}:
            return {"text": "  from object id  ", "company_name": "  Kept  "}
        return {"text": "wrong"}

    coll.find_one.side_effect = find_one
    text, company = asyncio.run(
        service._load_policy_text(
            str(oid),
            database="other-db",
            policy_collection="custom_policies",
        )
    )
    assert text == "from object id"
    assert company == "  Kept  "
    assert coll.find_one.call_count == 2


def test_load_policy_text_invalid_id_uses_raw_id_and_ignores_non_string_company() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "privacy-compliance", "policies", coll)

    def find_one(query: dict) -> dict | None:
        if query == {"document_id": "not-an-object-id"}:
            return None
        if query == {"_id": "not-an-object-id"}:
            return {"text": "raw id policy", "company_name": 12}
        raise AssertionError(query)

    coll.find_one.side_effect = find_one
    text, company = asyncio.run(service._load_policy_text("not-an-object-id"))
    assert text == "raw id policy"
    assert company is None


def test_load_policy_text_assembles_chunks_when_text_is_blank() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "privacy-compliance", "policies", coll)
    coll.find_one.return_value = {
        "text": "   ",
        "company_name": None,
        "policy_chunks": [
            "not-a-dict",
            {"chunk_text": "  first  "},
            {"chunk_text": ""},
            {"chunk_text": "   "},
            {"other": "skip"},
            {"chunk_text": "second"},
        ],
    }
    text, company = asyncio.run(service._load_policy_text("pol-1"))
    # Empty chunk_text is skipped. Whitespace-only chunk_text is truthy, so it
    # is kept as an empty segment and adds a blank line between neighbors.
    assert text == "first\n\n\n\nsecond"
    assert company is None
    coll.find_one.assert_called_once_with({"document_id": "pol-1"})


def test_load_policy_text_missing_document_is_empty() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "privacy-compliance", "policies", coll)
    coll.find_one.return_value = None
    assert asyncio.run(service._load_policy_text("missing")) == ("", None)


def test_load_policy_from_chunks_joins_header_and_first_string_company() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "privacy-compliance", "policy_legal_embeddings", coll)
    cursor = MagicMock()
    cursor.sort.return_value = [
        {
            "chunk_header_text": " H1 ",
            "chunk_text": " body ",
            "chunk_index": 2,
            "company_name": 5,
        },
        {
            "chunk_header_text": "Only header",
            "chunk_text": "  ",
            "company_name": None,
        },
        {"chunk_text": "plain", "company_name": "Beta"},
        {
            "chunk_header_text": "H0",
            "chunk_text": "zero",
            "company_name": "Ignored",
        },
    ]
    coll.find.return_value = cursor

    text, company = asyncio.run(
        service._load_policy_from_chunks("pol-9")
    )
    assert text == "H1\nbody\n\nplain\n\nH0\nzero"
    assert company == "Beta"
    coll.find.assert_called_once_with(
        {"document_id": "pol-9"},
        {
            "chunk_text": 1,
            "chunk_header_text": 1,
            "chunk_index": 1,
            "company_name": 1,
        },
    )
    cursor.sort.assert_called_once_with("chunk_index", 1)


def test_load_policy_from_chunks_empty_is_empty() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "archive", "chunks", coll)
    cursor = MagicMock()
    cursor.sort.return_value = []
    coll.find.return_value = cursor
    assert asyncio.run(
        service._load_policy_from_chunks("pol-9", database="archive", collection="chunks")
    ) == ("", None)


def test_fetch_v4_statute_docs_fans_out_blank_jurisdiction_and_aliases() -> None:
    service, mongo = _service()
    coll = MagicMock()
    _bind(mongo, "statutes-db", "statute_sub_topic_embeddings", coll)
    blank = {"jurisdiction": "", "header_text": "blank"}
    spaces = {"jurisdiction": "  ", "header_text": "spaces"}
    missing = {"header_text": "missing"}
    california = {"jurisdiction": "California", "header_text": "ccpa"}
    padded_ca = {"jurisdiction": "ca ", "header_text": "padded"}
    virginia = {"jurisdiction": "Virginia", "header_text": "vcdpa"}
    cursor = MagicMock()
    cursor.limit.return_value = [blank, spaces, missing, california, padded_ca, virginia]
    coll.find.return_value = cursor

    grouped = asyncio.run(service._fetch_v4_statute_docs(["CA", "VA"]))

    coll.find.assert_called_once_with(
        {"category": {"$in": ["consumer_rights", "controller_duties"]}}
    )
    cursor.limit.assert_called_once_with(200)
    # "California" and "ca" normalize to CA. "Virginia" is not in the
    # jurisdiction map, so it does not match a requested "VA".
    assert [doc["header_text"] for doc in grouped["CA"]] == [
        "blank",
        "spaces",
        "missing",
        "ccpa",
        "padded",
    ]
    assert [doc["header_text"] for doc in grouped["VA"]] == [
        "blank",
        "spaces",
        "missing",
    ]
    assert grouped["CA"][0] is grouped["VA"][0]

    cursor.limit.return_value = [virginia]
    grouped = asyncio.run(service._fetch_v4_statute_docs(["Virginia", "VA"]))
    assert [doc["header_text"] for doc in grouped["Virginia"]] == ["vcdpa"]
    assert grouped["VA"] == []


def test_fetch_v4_statute_docs_matches_request_spelling_and_custom_query() -> None:
    service, mongo = _service(statute_database="   ")
    assert service._retrieval_database == "privacy-compliance"
    coll = MagicMock()
    _bind(mongo, "privacy-compliance", "statute_sub_topic_embeddings", coll)
    cursor = MagicMock()
    cursor.limit.return_value = [{"jurisdiction": "CA", "header_text": "hit"}]
    coll.find.return_value = cursor

    grouped = asyncio.run(
        service._fetch_v4_statute_docs(
            ["california"],
            categories=["only_one"],
            limit=3,
        )
    )
    coll.find.assert_called_once_with({"category": {"$in": ["only_one"]}})
    cursor.limit.assert_called_once_with(3)
    assert list(grouped) == ["california"]
    assert grouped["california"][0]["header_text"] == "hit"


def test_v4_statute_text_prefers_subtopic_and_keeps_header() -> None:
    assert ComplianceSuiteService._v4_statute_text(
        {
            "header_text": " § 1798.100 ",
            "subtopic_text": " sub topic ",
            "requirement_summary": "summary",
        }
    ) == "§ 1798.100\n\nsub topic"
    assert ComplianceSuiteService._v4_statute_text(
        {"header_text": "  ", "requirement_summary": " summary only "}
    ) == "summary only"
    assert ComplianceSuiteService._v4_statute_text(
        {"header_text": " Header only ", "subtopic_text": "  "}
    ) == "Header only"
    assert ComplianceSuiteService._v4_statute_text({}) == ""
