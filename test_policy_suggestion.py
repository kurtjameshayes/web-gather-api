"""Regression tests for policy suggestion generation and service orchestration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import SuggestPolicyRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError
from llm_client import AnthropicLLMClient


@pytest.fixture
def anyio_backend() -> str:
    """Keep async tests independent of optional Trio installation."""
    return "asyncio"


@pytest.fixture
def suggestion_request() -> SuggestPolicyRequest:
    return SuggestPolicyRequest(
        policy_text="We retain account data indefinitely.",
        gap_analysis_text="The policy does not define a retention period.",
        gap_analysis_match="missing",
        statute_text="A controller must disclose its retention period.",
    )


@pytest.fixture
def suggestion_service() -> tuple[ComplianceSuiteService, MagicMock, MagicMock]:
    config = load_config()
    llm_client = MagicMock()
    llm_client.suggest_policy = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm_client,
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )
    return service, llm_client, rate_limiter


@pytest.mark.anyio
async def test_llm_suggest_policy_maps_inputs_and_normalizes_output() -> None:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._call_json = AsyncMock(
        return_value={
            "suggested_policy_text": "We retain account data for 30 days.",
            "modifications_description": None,
        }
    )

    result = await client.suggest_policy(
        policy_text="current-policy",
        gap_analysis_text="retention-gap",
        gap_analysis_match="partial",
        statute_text="retention-statute",
    )

    prompt = client._call_json.await_args.args[0]
    assert "current-policy" in prompt
    assert "retention-gap" in prompt
    assert "partial" in prompt
    assert "retention-statute" in prompt
    client._call_json.assert_awaited_once_with(prompt, max_tokens=4096)
    assert result == {
        "suggested_policy_text": "We retain account data for 30 days.",
        "modifications_description": "",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "llm_output",
    [
        None,
        {},
        {"suggested_policy_text": ["not", "text"]},
    ],
)
async def test_llm_suggest_policy_rejects_unusable_output(llm_output: object) -> None:
    client = AnthropicLLMClient.__new__(AnthropicLLMClient)
    client._call_json = AsyncMock(return_value=llm_output)

    result = await client.suggest_policy("policy", "gap", "missing", "statute")

    assert result is None


@pytest.mark.anyio
async def test_service_suggest_policy_delegates_complete_legal_context(
    suggestion_request: SuggestPolicyRequest,
    suggestion_service: tuple[ComplianceSuiteService, MagicMock, MagicMock],
) -> None:
    service, llm_client, rate_limiter = suggestion_service
    llm_client.suggest_policy.return_value = {
        "suggested_policy_text": "We retain account data for 30 days.",
        "modifications_description": "Added a defined retention period.",
    }

    response = await service.suggest_policy(suggestion_request)

    rate_limiter.allow.assert_awaited_once_with()
    llm_client.suggest_policy.assert_awaited_once_with(
        policy_text=suggestion_request.policy_text,
        gap_analysis_text=suggestion_request.gap_analysis_text,
        gap_analysis_match=suggestion_request.gap_analysis_match,
        statute_text=suggestion_request.statute_text,
    )
    assert response.suggested_policy_text == "We retain account data for 30 days."
    assert response.modifications_description == "Added a defined retention period."
    assert response.analyzed_at


@pytest.mark.anyio
async def test_service_suggest_policy_surfaces_generation_failure(
    suggestion_request: SuggestPolicyRequest,
    suggestion_service: tuple[ComplianceSuiteService, MagicMock, MagicMock],
) -> None:
    service, llm_client, _ = suggestion_service
    llm_client.suggest_policy.return_value = None

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.suggest_policy(suggestion_request)

    assert exc_info.value.status_code == 502
    assert "failed to generate policy suggestion" in str(exc_info.value)
