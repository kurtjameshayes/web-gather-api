"""Regression tests for ComplianceSuiteService consumer-rights router flow."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Keep imports lightweight/deterministic in unit tests.
sys.modules["anthropic"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService


def _build_service() -> tuple[ComplianceSuiteService, MagicMock, MagicMock]:
    config = load_config()
    mongo_client = MagicMock()
    retriever = MagicMock()
    llm_client = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    service = ComplianceSuiteService(
        mongo_client=mongo_client,
        config=config,
        retriever=retriever,
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
    )
    return service, llm_client, storage


def test_consumer_rights_router_handles_mixed_llm_outcomes_without_persistence() -> None:
    service, llm_client, storage = _build_service()
    service._fetch_v4_statute_docs = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "CA": [{"header_text": "CCPA §1798.130", "subtopic_text": "Right to know timeline"}],
            "NY": [{"header_text": "NYPA", "subtopic_text": "Hypothetical rights context"}],
        }
    )

    async def _llm_router_side_effect(**kwargs):
        request_type = kwargs["request_type"]
        if request_type == "deletion":
            return {
                "trees": {
                    "california": {
                        "id": "ca-del-1",
                        "action": "VERIFY AND PROCESS",
                        "detail": "Verify identity then process deletion request.",
                        "sla": "45 days (+ 45 day extension with notice)",
                        "exceptions": [],
                    }
                },
                "policy_gaps": {
                    "california": {"covered": False, "gap": "Policy omits deletion request timeline."},
                    "virginia": "invalid-shape-should-be-ignored",
                },
            }
        if request_type == "access":
            return None
        raise RuntimeError("simulated llm failure")

    llm_client.consumer_rights_router = AsyncMock(side_effect=_llm_router_side_effect)

    req = ConsumerRightsRouterRequest(
        text="We process consumer privacy requests via support portal.",
        applicable_jurisdictions=["CA", "NY"],
        request_types=["deletion", "access", "optout"],
        save_results=True,
    )
    response = asyncio.run(service.consumer_rights_router(req))

    assert response.policy_document_id is None
    assert response.company_name is None
    assert set(response.trees.keys()) == {"deletion", "access"}
    assert response.trees["access"] == {}
    assert response.policy_gaps["access"] == {}
    assert response.policy_gaps["deletion"]["california"].covered is False
    assert response.policy_gaps["deletion"]["california"].gap == "Policy omits deletion request timeline."
    assert "virginia" not in response.policy_gaps["deletion"]
    assert response.states["california"].abbr == "CA"
    assert response.states["ny"].name == "NY"

    # save_results=True should not persist when policy_document_id is absent (text mode).
    storage.write_compliance_result.assert_not_awaited()
    assert llm_client.consumer_rights_router.await_count == 3
    for call in llm_client.consumer_rights_router.await_args_list:
        assert call.kwargs["prompt_path"] == service._config.consumer_rights_router_prompt_path


def test_consumer_rights_router_persists_for_policy_document_id() -> None:
    service, llm_client, storage = _build_service()
    service._load_policy_from_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=("Policy text from chunks", "ACME Corp")
    )
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})  # type: ignore[method-assign]
    llm_client.consumer_rights_router = AsyncMock(
        return_value={
            "trees": {
                "california": {
                    "id": "ca-del-1",
                    "action": "PROCESS REQUEST",
                    "detail": "Handle verified request and respond.",
                    "sla": "45 days",
                    "exceptions": [],
                }
            },
            "policy_gaps": {"california": {"covered": True, "gap": None}},
        }
    )

    req = ConsumerRightsRouterRequest(
        policy_document_id="policy-123",
        applicable_jurisdictions=["CA"],
        request_types=["deletion"],
        save_results=True,
    )
    response = asyncio.run(service.consumer_rights_router(req))

    assert response.policy_document_id == "policy-123"
    assert response.company_name == "ACME Corp"
    assert response.request_types["deletion"] == "Right to Delete"
    assert response.policy_gaps["deletion"]["california"].covered is True
    service._load_policy_from_chunks.assert_awaited_once_with(
        "policy-123",
        database="privacy-compliance",
        collection="policy_legal_embeddings",
    )

    storage.write_compliance_result.assert_awaited_once()
    saved_doc = storage.write_compliance_result.await_args.args[0]
    assert saved_doc["result_type"] == "consumer_rights_router"
    assert saved_doc["policy_document_id"] == "policy-123"
