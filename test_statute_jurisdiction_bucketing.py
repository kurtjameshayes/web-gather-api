"""How citations and multi-jurisdictional comparison consume statute buckets.

The shared fetch copies unlabeled statutes into every requested jurisdiction.
Citations then dedupe those copies and only read 15 documents per jurisdiction.
Multi-jurisdictional comparison analyzes each copy. Requesting both a code and
its alias keeps the matches on the later label only.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import CitationsRequest, MultiJurisdictionalRequest
from compliance_suite_service import ComplianceSuiteService


POLICY_DB = "policy-db"
STATUTE_DB = "statute-db"
POLICY_TEXT = "We honor deletion requests."


def _mongo(
    statute_docs: list[dict[str, Any]],
    policy_doc: dict[str, Any] | None,
) -> MagicMock:
    statute_coll = MagicMock(name="statute_coll")
    cursor = MagicMock(name="statute_cursor")
    cursor.limit.return_value = statute_docs
    statute_coll.find.return_value = cursor

    policy_coll = MagicMock(name="policy_coll")
    policy_coll.find_one.return_value = policy_doc

    def get_db(name: str) -> MagicMock:
        db = MagicMock(name=f"db:{name}")

        def get_coll(coll_name: str, db_name: str = name) -> MagicMock:
            if db_name == STATUTE_DB and coll_name == "statute_sub_topic_embeddings":
                return statute_coll
            if db_name == POLICY_DB and coll_name == "policies":
                return policy_coll
            empty = MagicMock(name=f"empty:{db_name}.{coll_name}")
            empty.find_one.return_value = None
            empty_cursor = MagicMock()
            empty_cursor.limit.return_value = []
            empty.find.return_value = empty_cursor
            return empty

        db.__getitem__.side_effect = get_coll
        return db

    mongo = MagicMock(name="mongo")
    mongo.__getitem__.side_effect = get_db
    return mongo


def _service(
    mongo: MagicMock,
    llm_calls: list[dict[str, Any]],
    *,
    default_jurisdictions: list[str] | None = None,
) -> ComplianceSuiteService:
    config = load_config()
    config.statute_database = STATUTE_DB
    config.compliance_database = POLICY_DB
    config.policies_collection = "policies"
    config.statute_sub_topic_embeddings_collection = "statute_sub_topic_embeddings"
    config.statute_jurisdiction_field = "jurisdiction"
    config.policy_document_id_field = "document_id"
    config.default_jurisdictions = ["CA"] if default_jurisdictions is None else default_jurisdictions
    config.llm_concurrency = 1

    rate_limiter = MagicMock()

    async def _allow() -> bool:
        return True

    rate_limiter.allow = _allow

    llm = MagicMock()

    async def citation_check(**kwargs: Any) -> dict[str, Any]:
        llm_calls.append(kwargs)
        return {"alignment": True, "statute_excerpt": "kept"}

    llm.citation_check = citation_check
    return ComplianceSuiteService(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )


def _cited(llm_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(call["jurisdiction"], call["statute_reference"]) for call in llm_calls]


def test_citations_keep_shared_statutes_on_the_first_jurisdiction_only() -> None:
    """Missing, whitespace, and numeric-zero jurisdictions are queued for every state.

    The citation dedupe key ignores jurisdiction, so those shared rows are sent
    to the model only for CA. A stored California row follows CA. Virginia does
    not follow VA, while the codes VA and va do.
    """
    docs = [
        {"document_id": "blank", "subtopic_text": "UNLABELED RULE"},
        {"document_id": "ws", "jurisdiction": "   ", "subtopic_text": "WHITESPACE RULE"},
        {"document_id": "zero", "jurisdiction": 0, "subtopic_text": "ZERO RULE"},
        {"document_id": "ca-alias", "jurisdiction": "California", "subtopic_text": "CA ONLY"},
        {"document_id": "va-name", "jurisdiction": "Virginia", "subtopic_text": "VIRGINIA NAME"},
        {"document_id": "va", "jurisdiction": "VA", "subtopic_text": "VA ONLY"},
        {"document_id": "va-lower", "jurisdiction": "va", "subtopic_text": "VA LOWER"},
    ]
    mongo = _mongo(docs, {"text": POLICY_TEXT, "company_name": "Acme"})
    llm_calls: list[dict[str, Any]] = []
    service = _service(mongo, llm_calls)
    result = asyncio.run(
        service.citations(
            CitationsRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA", "VA"])
        )
    )

    assert _cited(llm_calls) == [
        ("CA", "blank"),
        ("CA", "ws"),
        ("CA", "zero"),
        ("CA", "ca-alias"),
        ("VA", "va"),
        ("VA", "va-lower"),
    ]
    assert "VIRGINIA NAME" not in {call["statute_chunk_text"] for call in llm_calls}
    assert result.applicable_jurisdictions == ["CA", "VA"]
    assert [(item.jurisdiction, item.statute_reference) for item in result.citations] == _cited(llm_calls)
    assert result.summary.total_citations == 6
    assert result.summary.aligned == 6
    assert result.summary.not_aligned == 0


def test_unlabeled_rows_fill_each_citation_window() -> None:
    """Fifteen unlabeled statutes occupy the 15-document window for CA and for VA.

    The Virginia statute stored after them is never cited.
    """
    docs = [
        {"document_id": f"fill-{i:02d}", "subtopic_text": f"FILL {i:02d}"}
        for i in range(15)
    ]
    docs.append({"document_id": "va-real", "jurisdiction": "VA", "subtopic_text": "REAL VA"})
    mongo = _mongo(docs, {"text": POLICY_TEXT, "company_name": "Acme"})
    llm_calls: list[dict[str, Any]] = []
    service = _service(mongo, llm_calls)
    result = asyncio.run(
        service.citations(
            CitationsRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA", "VA"])
        )
    )

    assert _cited(llm_calls) == [("CA", f"fill-{i:02d}") for i in range(15)]
    assert all(call["statute_chunk_text"] != "REAL VA" for call in llm_calls)
    assert result.summary.total_citations == 15
    assert [item.jurisdiction for item in result.citations] == ["CA"] * 15


def test_requesting_a_code_and_its_alias_keeps_the_later_label() -> None:
    """CA and California collapse to one bucket, and the later request label wins.

    Both stored spellings are cited once, under California. Texas is not.
    """
    docs = [
        {"document_id": "code", "jurisdiction": "CA", "subtopic_text": "CA CODE"},
        {"document_id": "name", "jurisdiction": "california", "subtopic_text": "CA NAME"},
        {"document_id": "tx", "jurisdiction": "Texas", "subtopic_text": "TEXAS ONLY"},
    ]
    mongo = _mongo(docs, {"text": POLICY_TEXT, "company_name": "Acme"})
    llm_calls: list[dict[str, Any]] = []
    service = _service(mongo, llm_calls)
    result = asyncio.run(
        service.citations(
            CitationsRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA", "California"],
            )
        )
    )

    assert _cited(llm_calls) == [("California", "code"), ("California", "name")]
    assert result.applicable_jurisdictions == ["CA", "California"]
    assert [item.jurisdiction for item in result.citations] == ["California", "California"]


def test_multi_jurisdictional_repeats_unlabeled_statutes() -> None:
    """Multi-jurisdictional comparison has no cross-state citation dedupe.

    The unlabeled statute is extracted for CA and again for VA.
    """
    docs = [
        {"document_id": "shared", "subtopic_text": "SHARED RULE"},
        {"document_id": "ca", "jurisdiction": "California", "subtopic_text": "CA ONLY"},
        {"document_id": "va", "jurisdiction": "VA", "subtopic_text": "VA ONLY"},
        {"document_id": "va-name", "jurisdiction": "Virginia", "subtopic_text": "VIRGINIA NAME"},
    ]
    mongo = _mongo(docs, None)
    service = _service(mongo, [])
    extracted: list[str] = []

    async def requirement_extraction(chunk_text: str) -> dict[str, Any]:
        extracted.append(chunk_text)
        return {"requirements": []}

    service._llm_client.requirement_extraction = requirement_extraction
    result = asyncio.run(
        service.multi_jurisdictional(
            MultiJurisdictionalRequest(applicable_jurisdictions=["CA", "VA"])
        )
    )

    assert extracted == ["SHARED RULE", "CA ONLY", "SHARED RULE", "VA ONLY"]
    assert result.applicable_jurisdictions == ["CA", "VA"]
    assert result.strictest_common_denominator == []


def test_citations_do_not_invent_california_when_defaults_are_empty() -> None:
    """An omitted jurisdiction list does not gain the gap-analysis CA fallback."""
    docs = [
        {"document_id": "ca", "jurisdiction": "California", "subtopic_text": "CA RULE"},
        {"document_id": "blank", "subtopic_text": "UNLABELED RULE"},
    ]
    mongo = _mongo(docs, {"text": POLICY_TEXT, "company_name": "Acme"})
    llm_calls: list[dict[str, Any]] = []
    service = _service(mongo, llm_calls, default_jurisdictions=[])
    result = asyncio.run(service.citations(CitationsRequest(policy_document_id="pol-1")))

    assert llm_calls == []
    assert result.applicable_jurisdictions == []
    assert result.citations == []
    assert result.company_name == "Acme"
    assert result.summary.total_citations == 0


def test_truthy_numeric_jurisdiction_raises_before_the_model_call() -> None:
    mongo = _mongo(
        [{"document_id": "num", "jurisdiction": 1, "subtopic_text": "NUMERIC RULE"}],
        {"text": POLICY_TEXT, "company_name": "Acme"},
    )
    llm_calls: list[dict[str, Any]] = []
    service = _service(mongo, llm_calls)

    with pytest.raises(AttributeError, match="strip"):
        asyncio.run(
            service.citations(
                CitationsRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA"])
            )
        )

    assert llm_calls == []
