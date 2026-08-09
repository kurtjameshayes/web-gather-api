"""Regression tests for ComplianceSuiteService._load_policy_from_chunks assembly."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

# Lightweight stubs for optional heavy imports pulled in by compliance_suite_service.
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_suite_service import ComplianceSuiteService


def _run(coro):
    return asyncio.run(coro)


def _make_service(chunks: List[dict], doc_id_field: str = "document_id") -> ComplianceSuiteService:
    """Build a minimal service instance without running the heavy constructor."""
    service = ComplianceSuiteService.__new__(ComplianceSuiteService)
    service._config = SimpleNamespace(policy_document_id_field=doc_id_field)

    coll = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = chunks
    coll.find.return_value = cursor

    db = MagicMock()
    db.__getitem__.return_value = coll
    mongo = MagicMock()
    mongo.__getitem__.return_value = db
    service._mongo_client = mongo
    service._chunks_coll = coll  # for assertions
    return service


def test_load_policy_from_chunks_orders_and_joins_header_body() -> None:
    chunks = [
        {
            "chunk_index": 0,
            "chunk_header_text": "1. Collection",
            "chunk_text": "We collect personal data.",
            "company_name": "Acme Corp",
        },
        {
            "chunk_index": 1,
            "chunk_header_text": "  ",
            "chunk_text": "We retain data for 12 months.",
        },
        {
            "chunk_index": 2,
            "chunk_header_text": "3. Sharing",
            "chunk_text": "",
        },
    ]
    service = _make_service(chunks)

    text, company = _run(service._load_policy_from_chunks("pol-1"))

    assert company == "Acme Corp"
    assert text == (
        "1. Collection\nWe collect personal data.\n\n"
        "We retain data for 12 months."
    )
    service._chunks_coll.find.assert_called_once_with(
        {"document_id": "pol-1"},
        {"chunk_text": 1, "chunk_header_text": 1, "chunk_index": 1, "company_name": 1},
    )
    service._chunks_coll.find.return_value.sort.assert_called_once_with("chunk_index", 1)


def test_load_policy_from_chunks_empty_returns_blank() -> None:
    service = _make_service([])

    text, company = _run(service._load_policy_from_chunks("missing"))

    assert text == ""
    assert company is None


def test_load_policy_from_chunks_uses_configured_document_id_field() -> None:
    chunks = [
        {
            "chunk_index": 0,
            "chunk_text": "Only body text",
            "company_name": 123,  # non-string ignored
        },
        {
            "chunk_index": 1,
            "chunk_text": "Second",
            "company_name": "Beta Inc",
        },
    ]
    service = _make_service(chunks, doc_id_field="policy_id")

    text, company = _run(
        service._load_policy_from_chunks(
            "p-9",
            database="custom-db",
            collection="custom_embeddings",
        )
    )

    assert text == "Only body text\n\nSecond"
    assert company == "Beta Inc"
    service._mongo_client.__getitem__.assert_called_with("custom-db")
    service._mongo_client.__getitem__.return_value.__getitem__.assert_called_with(
        "custom_embeddings"
    )
    service._chunks_coll.find.assert_called_once_with(
        {"policy_id": "p-9"},
        {"chunk_text": 1, "chunk_header_text": 1, "chunk_index": 1, "company_name": 1},
    )
