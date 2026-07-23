"""Regression tests for compliance applicability orchestration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import ApplicabilityRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def service() -> ComplianceSuiteService:
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=load_config(),
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=MagicMock(),
    )


@pytest.mark.anyio
async def test_applicability_uses_trimmed_inline_text(service: ComplianceSuiteService) -> None:
    service._rate_limiter.allow = AsyncMock(return_value=True)
    service._llm_client.applicability = AsyncMock(
        return_value={
            "applicable_jurisdictions": ["California", "Colorado"],
            "confidence": {"California": 0.98, "Colorado": 0.82},
        }
    )
    service._load_policy_text = AsyncMock()

    result = await service.applicability(
        ApplicabilityRequest(text="  We collect California resident data.  ")
    )

    service._load_policy_text.assert_not_awaited()
    service._llm_client.applicability.assert_awaited_once_with(
        "We collect California resident data."
    )
    assert result.applicable_jurisdictions == ["California", "Colorado"]
    assert result.confidence == {"California": 0.98, "Colorado": 0.82}


@pytest.mark.anyio
async def test_applicability_loads_requested_policy_source(service: ComplianceSuiteService) -> None:
    service._rate_limiter.allow = AsyncMock(return_value=True)
    service._load_policy_text = AsyncMock(return_value=("Stored policy text", "Acme"))
    service._llm_client.applicability = AsyncMock(
        return_value={"applicable_jurisdictions": None, "confidence": None}
    )

    result = await service.applicability(
        ApplicabilityRequest(
            policy_document_id="policy-123",
            database="tenant-db",
            policy_collection="published-policies",
        )
    )

    service._load_policy_text.assert_awaited_once_with(
        "policy-123",
        database="tenant-db",
        policy_collection="published-policies",
    )
    service._llm_client.applicability.assert_awaited_once_with("Stored policy text")
    assert result.applicable_jurisdictions == []
    assert result.confidence is None


@pytest.mark.anyio
async def test_applicability_rejects_empty_stored_policy(service: ComplianceSuiteService) -> None:
    service._rate_limiter.allow = AsyncMock(return_value=True)
    service._load_policy_text = AsyncMock(return_value=("", None))
    service._llm_client.applicability = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.applicability(ApplicabilityRequest(policy_document_id="missing-policy"))

    assert exc_info.value.status_code == 400
    assert str(exc_info.value) == "Policy text not found or empty."
    service._llm_client.applicability.assert_not_awaited()


@pytest.mark.anyio
async def test_applicability_rate_limit_short_circuits_work(
    service: ComplianceSuiteService,
) -> None:
    service._rate_limiter.allow = AsyncMock(return_value=False)
    service._load_policy_text = AsyncMock()
    service._llm_client.applicability = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.applicability(ApplicabilityRequest(policy_document_id="policy-123"))

    assert exc_info.value.status_code == 429
    assert str(exc_info.value) == "Rate limit exceeded"
    service._load_policy_text.assert_not_awaited()
    service._llm_client.applicability.assert_not_awaited()
