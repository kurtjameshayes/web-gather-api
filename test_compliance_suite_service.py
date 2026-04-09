"""Regression tests for ComplianceSuiteService consumer-rights router flow."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Avoid loading heavy optional dependencies during unit tests.
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _build_service() -> tuple[ComplianceSuiteService, MagicMock, MagicMock]:
    config = load_config()
    config.auth_required = False
    config.default_jurisdictions = ["CA", "VA"]

    llm_client = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
    )
    return service, llm_client, storage


def test_consumer_rights_router_handles_partial_llm_failures_with_text_input() -> None:
    service, llm_client, storage = _build_service()
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {
                    "header_text": "CCPA 1798.105",
                    "subtopic_text": "Delete request handling",
                }
            ],
            "ZZ": [],
        }
    )
    llm_client.consumer_rights_router = AsyncMock(
        side_effect=[
            {
                "trees": {"california": {"id": "ca-del-1", "action": "VERIFY IDENTITY"}},
                "policy_gaps": {
                    "california": {"covered": 1, "gap": None},
                    "ignored_non_dict": "skip",
                },
            },
            RuntimeError("upstream timeout"),
        ]
    )

    req = ConsumerRightsRouterRequest(
        text="  We process deletion requests within 45 days.  ",
        applicable_jurisdictions=["CA", "ZZ"],
        request_types=["deletion", "access"],
        save_results=True,
    )

    result = asyncio.run(service.consumer_rights_router(req))

    assert result.policy_document_id is None
    assert result.company_name is None
    assert result.applicable_jurisdictions == ["CA", "ZZ"]
    assert set(result.states.keys()) == {"california", "zz"}
    assert result.request_types["deletion"] == "Right to Delete"
    assert result.request_types["access"] == "Right to Access / Know"
    assert result.trees["deletion"]["california"]["id"] == "ca-del-1"
    assert "access" not in result.trees
    assert result.policy_gaps["deletion"]["california"].covered is True
    assert result.policy_gaps["deletion"]["california"].gap is None

    service._fetch_v4_statute_docs.assert_awaited_once_with(
        ["CA", "ZZ"], categories=["consumer_rights"]
    )
    assert llm_client.consumer_rights_router.await_count == 2
    first_call = llm_client.consumer_rights_router.await_args_list[0].kwargs
    assert first_call["prompt_path"] == service._config.consumer_rights_router_prompt_path
    assert "- CA: California (CCPA/CPRA) (slug: california)" in first_call["jurisdictions"]
    assert "- ZZ: ZZ (slug: zz)" in first_call["jurisdictions"]
    storage.write_compliance_result.assert_not_called()


def test_consumer_rights_router_loads_chunks_and_persists_when_document_id_present() -> None:
    service, llm_client, storage = _build_service()
    service._load_policy_from_chunks = AsyncMock(return_value=("Policy body", "Acme Inc"))
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})
    llm_client.consumer_rights_router = AsyncMock(return_value=None)

    req = ConsumerRightsRouterRequest(
        policy_document_id="doc-123",
        applicable_jurisdictions=["CA"],
        request_types=["optout"],
        save_results=True,
    )

    result = asyncio.run(service.consumer_rights_router(req))

    assert result.policy_document_id == "doc-123"
    assert result.company_name == "Acme Inc"
    assert result.trees["optout"] == {}
    assert result.policy_gaps["optout"] == {}

    service._load_policy_from_chunks.assert_awaited_once_with(
        "doc-123",
        database="privacy-compliance",
        collection="policy_legal_embeddings",
    )
    storage.write_compliance_result.assert_awaited_once()
    written_doc = storage.write_compliance_result.await_args.args[0]
    assert written_doc["result_type"] == "consumer_rights_router"
    assert written_doc["policy_document_id"] == "doc-123"


def test_consumer_rights_router_errors_when_loaded_policy_is_empty() -> None:
    service, llm_client, storage = _build_service()
    service._load_policy_from_chunks = AsyncMock(return_value=("", None))
    service._fetch_v4_statute_docs = AsyncMock()
    llm_client.consumer_rights_router = AsyncMock()

    req = ConsumerRightsRouterRequest(policy_document_id="missing-doc")

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        asyncio.run(service.consumer_rights_router(req))

    assert exc_info.value.status_code == 400
    assert "not found or empty" in str(exc_info.value)
    service._fetch_v4_statute_docs.assert_not_awaited()
    llm_client.consumer_rights_router.assert_not_awaited()
    storage.write_compliance_result.assert_not_called()
