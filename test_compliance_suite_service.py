"""Regression tests for ComplianceSuiteService consumer-rights routing flow."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Keep tests lightweight when anthropic SDK is unavailable.
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


@pytest.fixture
def service() -> ComplianceSuiteService:
    """Build a ComplianceSuiteService with mocked external dependencies."""
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    svc = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=load_config(),
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )
    svc._storage.write_compliance_result = AsyncMock()
    return svc


def test_consumer_rights_router_handles_mixed_llm_outcomes(service: ComplianceSuiteService) -> None:
    """Partial LLM failures should not break the aggregate response payload."""

    async def _llm_router(**kwargs):
        request_type = kwargs["request_type"]
        if request_type == "access":
            return {
                "trees": {
                    "california": {
                        "id": "ca-access",
                        "action": "verify-identity-and-respond",
                    }
                },
                "policy_gaps": {
                    "california": {"covered": True, "gap": None},
                    "ny": {"covered": False, "gap": "No response timeline specified."},
                },
            }
        if request_type == "deletion":
            return None
        if request_type == "optout":
            raise RuntimeError("simulated llm failure")
        raise AssertionError(f"Unexpected request_type in test: {request_type}")

    service._llm_client.consumer_rights_router = AsyncMock(side_effect=_llm_router)
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [{"header_text": "CCPA §1798.100", "subtopic_text": "Access right details."}],
            "NY": [],
        }
    )

    response = asyncio.run(
        service.consumer_rights_router(
            ConsumerRightsRouterRequest(
                text="Privacy policy text",
                applicable_jurisdictions=["CA", "NY"],
                request_types=["access", "deletion", "optout"],
            )
        )
    )

    assert set(response.request_types.keys()) == {"access", "deletion", "optout"}
    assert response.states["california"].abbr == "CA"
    assert response.states["ny"].name == "NY"

    assert "access" in response.trees
    assert response.trees["access"]["california"]["action"] == "verify-identity-and-respond"
    assert response.policy_gaps["access"]["california"].covered is True

    assert response.trees["deletion"] == {}
    assert response.policy_gaps["deletion"] == {}

    # Exceptions are skipped and do not poison the full response.
    assert "optout" not in response.trees
    service._storage.write_compliance_result.assert_not_called()


def test_consumer_rights_router_persists_result_for_policy_id(service: ComplianceSuiteService) -> None:
    """When save_results=True with a policy_document_id, result should be persisted."""
    service._load_policy_from_chunks = AsyncMock(return_value=("Policy body text", "Acme Inc."))
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})
    service._llm_client.consumer_rights_router = AsyncMock(
        return_value={
            "trees": {"california": {"id": "ca-delete", "action": "complete-deletion-workflow"}},
            "policy_gaps": {"california": {"covered": False, "gap": "No deletion SLA disclosed."}},
        }
    )

    response = asyncio.run(
        service.consumer_rights_router(
            ConsumerRightsRouterRequest(
                policy_document_id="pol-123",
                applicable_jurisdictions=["CA"],
                request_types=["deletion"],
                save_results=True,
            )
        )
    )

    assert response.policy_document_id == "pol-123"
    assert response.company_name == "Acme Inc."

    service._storage.write_compliance_result.assert_awaited_once()
    persisted_doc = service._storage.write_compliance_result.await_args.args[0]
    assert persisted_doc["result_type"] == "consumer_rights_router"
    assert persisted_doc["policy_document_id"] == "pol-123"
    assert "deletion" in persisted_doc["trees"]


def test_consumer_rights_router_rate_limit_exceeded(service: ComplianceSuiteService) -> None:
    """Rate-limit rejection should short-circuit before downstream calls."""
    service._rate_limiter.allow = AsyncMock(return_value=False)

    with pytest.raises(ComplianceSuiteServiceError) as exc:
        asyncio.run(
            service.consumer_rights_router(
                ConsumerRightsRouterRequest(text="Privacy policy text")
            )
        )

    assert exc.value.status_code == 429
    service._llm_client.consumer_rights_router.assert_not_called()
