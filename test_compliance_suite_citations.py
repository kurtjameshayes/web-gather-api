"""Service-level regression tests for compliance citation extraction."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from compliance_config import load_config
from compliance_suite_schemas import CitationsRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _build_service(*, rate_allowed: bool = True) -> tuple[ComplianceSuiteService, MagicMock]:
    config = load_config()
    config.default_jurisdictions = ["CA", "VA"]

    llm_client = MagicMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=rate_allowed)

    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm_client,
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )
    return service, llm_client


@pytest.mark.anyio
async def test_citations_deduplicates_statutes_and_summarizes_alignment() -> None:
    service, llm_client = _build_service()
    policy_text = "P" * 3500
    service._load_policy_text = AsyncMock(return_value=(policy_text, "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {
                    "document_id": "ca-delete",
                    "header_text": "§ 1798.105",
                    "subtopic_text": "Consumers may request deletion.",
                },
                {
                    "document_id": "ca-delete",
                    "header_text": "§ 1798.105",
                    "subtopic_text": "Consumers may request deletion.",
                },
                {"document_id": "empty"},
            ],
            "VA": [
                {
                    "_id": "va-access",
                    "requirement_summary": "Consumers may request access.",
                }
            ],
        }
    )
    llm_client.citation_check = AsyncMock(
        side_effect=[
            {"alignment": True, "statute_excerpt": "request deletion"},
            {"alignment": False, "statute_excerpt": "request access"},
        ]
    )

    result = await service.citations(CitationsRequest(policy_document_id="policy-1"))

    service._fetch_v4_statute_docs.assert_awaited_once_with(["CA", "VA"])
    llm_client.citation_check.assert_has_awaits(
        [
            call(
                policy_excerpt=policy_text[:3000],
                statute_chunk_text="§ 1798.105\n\nConsumers may request deletion.",
                statute_reference="ca-delete",
                jurisdiction="CA",
            ),
            call(
                policy_excerpt=policy_text[:3000],
                statute_chunk_text="Consumers may request access.",
                statute_reference="va-access",
                jurisdiction="VA",
            ),
        ]
    )
    assert result.company_name == "Acme"
    assert [citation.alignment for citation in result.citations] == [True, False]
    assert [citation.policy_excerpt for citation in result.citations] == [policy_text[:200]] * 2
    assert result.summary.model_dump() == {
        "total_citations": 2,
        "aligned": 1,
        "not_aligned": 1,
    }


@pytest.mark.anyio
async def test_citations_rate_limit_short_circuits_policy_and_llm_work() -> None:
    service, llm_client = _build_service(rate_allowed=False)
    service._load_policy_text = AsyncMock()
    service._fetch_v4_statute_docs = AsyncMock()
    llm_client.citation_check = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.citations(CitationsRequest(policy_document_id="policy-1"))

    assert exc_info.value.status_code == 429
    assert str(exc_info.value) == "Rate limit exceeded"
    service._load_policy_text.assert_not_awaited()
    service._fetch_v4_statute_docs.assert_not_awaited()
    llm_client.citation_check.assert_not_awaited()
