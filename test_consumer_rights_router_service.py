"""Regression tests for consumer-rights router limits that earlier suite tests left open.

Covers statute-context slicing, the save_results gate on a document id, and
rate-limit ordering. Does not retest text-only persistence, chunk-collection
paths, unknown-jurisdiction slugs, omitted request-type fan-out, covered "false",
or the undefined-logger NameError.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


class _Limiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls = 0

    async def allow(self) -> bool:
        self.calls += 1
        return self.allowed


class _Storage:
    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    async def write_compliance_result(self, doc: dict[str, Any]) -> str:
        self.docs.append(doc)
        return "stored"


class _LLM:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else {"trees": {}, "policy_gaps": {}}
        self.calls: list[dict[str, Any]] = []

    async def consumer_rights_router(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.payload


def _config():
    config = load_config()
    config.compliance_database = "privacy-compliance"
    config.statute_database = ""
    config.default_jurisdictions = ["CA", "VA", "CO", "CT"]
    config.consumer_rights_router_prompt_path = "prompts/consumer_rights_router.yaml"
    config.llm_concurrency = 4
    return config


def _service(llm: _LLM, storage: _Storage, limiter: _Limiter) -> ComplianceSuiteService:
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=_config(),
        retriever=MagicMock(),
        llm_client=llm,
        storage=storage,
        rate_limiter=limiter,
    )


async def _rate_limit_precedes_whitespace_text() -> None:
    llm = _LLM()
    fetch_calls: list[Any] = []

    async def fetch(*args, **kwargs):
        fetch_calls.append((args, kwargs))
        return {}

    service = _service(llm, _Storage(), _Limiter(allowed=False))
    service._fetch_v4_statute_docs = fetch  # type: ignore[method-assign]

    try:
        await service.consumer_rights_router(
            ConsumerRightsRouterRequest(text="   ", request_types=["deletion"])
        )
    except ComplianceSuiteServiceError as exc:
        assert exc.status_code == 429
        assert str(exc) == "Rate limit exceeded"
    else:
        raise AssertionError("expected rate limit")
    assert llm.calls == []
    assert fetch_calls == []

    service._rate_limiter = _Limiter(allowed=True)
    try:
        await service.consumer_rights_router(
            ConsumerRightsRouterRequest(text="  \n  ", request_types=["deletion"])
        )
    except ComplianceSuiteServiceError as exc:
        assert exc.status_code == 400
        assert str(exc) == "Policy text not found or empty."
    else:
        raise AssertionError("expected empty policy text")
    assert llm.calls == []
    assert fetch_calls == []


async def _statute_context_slices_and_coerces_gaps() -> None:
    long_text = "R" * 700
    docs = [{"header_text": "H", "subtopic_text": long_text} for _ in range(10)]
    docs.append({"subtopic_text": "ELEVENTH-MARKER"})
    payload = {
        "trees": {"california": {"id": "ask"}},
        "policy_gaps": {
            "california": {"covered": 0, "gap": "numeric zero"},
            "virginia": {"gap": "omitted covered"},
        },
    }
    llm = _LLM(payload)
    service = _service(llm, _Storage(), _Limiter())
    seen: dict[str, Any] = {}

    async def fetch(jurisdictions, categories=None, limit=200):
        seen["jurisdictions"] = list(jurisdictions)
        seen["categories"] = categories
        seen["limit"] = limit
        return {j: list(docs) for j in jurisdictions}

    service._fetch_v4_statute_docs = fetch  # type: ignore[method-assign]

    response = await service.consumer_rights_router(
        ConsumerRightsRouterRequest(
            text="Policy text.",
            applicable_jurisdictions=["CA", "VA"],
            request_types=["deletion"],
            save_results=False,
        )
    )

    assert seen == {
        "jurisdictions": ["CA", "VA"],
        "categories": ["consumer_rights"],
        "limit": 200,
    }
    context = llm.calls[0]["statute_context"]
    piece = "H\n\n" + ("R" * 700)
    assert len(context) == 8000
    assert "ELEVENTH-MARKER" not in context
    assert context.count("[CA] ") == 10
    assert context.count("[VA] ") == 4
    assert context.startswith("[CA] " + piece[:600])
    assert llm.calls[0]["policy_text"] == "Policy text."
    assert llm.calls[0]["request_type"] == "deletion"
    assert llm.calls[0]["request_type_label"] == "Right to Delete"
    assert llm.calls[0]["prompt_path"] == "prompts/consumer_rights_router.yaml"
    gaps = response.policy_gaps["deletion"]
    assert gaps["california"].covered is False
    assert gaps["california"].gap == "numeric zero"
    assert gaps["virginia"].covered is False
    assert gaps["virginia"].gap == "omitted covered"

    llm.calls.clear()
    seen.clear()
    await service.consumer_rights_router(
        ConsumerRightsRouterRequest(text="Policy text.", request_types=["access"])
    )
    assert seen["jurisdictions"] == ["CA", "VA", "CO", "CT"]
    assert llm.calls[0]["request_type_label"] == "Right to Access / Know"
    assert llm.calls[0]["jurisdictions"] == (
        "- CA: California (CCPA/CPRA) (slug: california)\n"
        "- VA: Virginia (VCDPA) (slug: virginia)\n"
        "- CO: Colorado (CPA) (slug: colorado)\n"
        "- CT: Connecticut (CTDPA) (slug: connecticut)"
    )


async def _save_results_requires_document_id() -> None:
    llm = _LLM({"trees": {"california": {"id": "tree"}}, "policy_gaps": {}})
    storage = _Storage()
    service = _service(llm, storage, _Limiter())

    async def load(policy_document_id, database="privacy-compliance", collection="policy_legal_embeddings"):
        assert policy_document_id == "pol-9"
        return "We delete data.", "Acme"

    async def fetch(jurisdictions, categories=None, limit=200):
        return {j: [] for j in jurisdictions}

    service._load_policy_from_chunks = load  # type: ignore[method-assign]
    service._fetch_v4_statute_docs = fetch  # type: ignore[method-assign]

    skipped = await service.consumer_rights_router(
        ConsumerRightsRouterRequest(
            policy_document_id="pol-9",
            applicable_jurisdictions=["CA"],
            request_types=["deletion"],
            save_results=False,
        )
    )
    assert skipped.policy_document_id == "pol-9"
    assert skipped.company_name == "Acme"
    assert storage.docs == []

    saved = await service.consumer_rights_router(
        ConsumerRightsRouterRequest(
            policy_document_id="pol-9",
            applicable_jurisdictions=["CA"],
            request_types=["deletion"],
            save_results=True,
        )
    )
    assert saved.trees["deletion"]["california"]["id"] == "tree"
    assert len(storage.docs) == 1
    stored = storage.docs[0]
    assert stored["result_type"] == "consumer_rights_router"
    assert stored["policy_document_id"] == "pol-9"
    assert stored["company_name"] == "Acme"
    assert stored["trees"]["deletion"]["california"]["id"] == "tree"
    assert "analyzed_at" in stored


def test_rate_limit_precedes_whitespace_text() -> None:
    asyncio.run(_rate_limit_precedes_whitespace_text())


def test_statute_context_slices_and_coerces_gaps() -> None:
    asyncio.run(_statute_context_slices_and_coerces_gaps())


def test_save_results_requires_document_id() -> None:
    asyncio.run(_save_results_requires_document_id())
