"""Focused regression tests for compliance risk-assessment orchestration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import RiskAssessmentRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _service() -> ComplianceSuiteService:
    config = load_config()
    config.default_jurisdictions = ["CA", "VA"]
    llm_client = MagicMock()
    llm_client.risk_assessment = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm_client,
        storage=MagicMock(),
        rate_limiter=rate_limiter,
    )


@pytest.mark.anyio
async def test_risk_assessment_builds_statute_context_and_markdown_report() -> None:
    """The service sends jurisdiction-tagged statutes to the LLM and renders its findings."""
    service = _service()
    service._load_policy_text = AsyncMock(return_value=("We disclose collected data.", "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {
                    "header_text": "Cal. Civ. Code § 1798.100",
                    "subtopic_text": "Disclose the categories of personal information collected.",
                }
            ],
            "VA": [
                {
                    "requirement_summary": "Provide a right to delete personal data.",
                }
            ],
        }
    )
    assessment = {
        "processing_purposes": ["Provide the service"],
        "data_categories": ["Contact information"],
        "risks": [
            {"description": "Over-retention", "severity": "high"},
            "Unclear deletion process",
        ],
        "mitigations": ["Publish a retention schedule"],
        "gaps_from_statute": [],
    }
    service._llm_client.risk_assessment.return_value = assessment

    result = await service.risk_assessment(
        RiskAssessmentRequest(
            policy_document_id="policy-1",
            template_id="dpia",
            include_report=True,
        )
    )

    service._fetch_v4_statute_docs.assert_awaited_once_with(["CA", "VA"])
    service._llm_client.risk_assessment.assert_awaited_once_with(
        "We disclose collected data.",
        "[CA] Cal. Civ. Code § 1798.100\n\n"
        "Disclose the categories of personal information collected.\n\n"
        "[VA] Provide a right to delete personal data.",
    )
    assert result.policy_document_id == "policy-1"
    assert result.company_name == "Acme"
    assert result.applicable_jurisdictions == ["CA", "VA"]
    assert result.template_id == "dpia"
    assert result.assessment == assessment
    assert result.report is not None
    assert "- Over-retention (severity: high)" in result.report
    assert "- Unclear deletion process" in result.report
    assert "- Publish a retention schedule" in result.report


@pytest.mark.anyio
async def test_risk_assessment_missing_policy_stops_before_retrieval_and_llm() -> None:
    """An absent policy fails clearly without spending retrieval or LLM resources."""
    service = _service()
    service._load_policy_text = AsyncMock(return_value=("", None))
    service._fetch_v4_statute_docs = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.risk_assessment(
            RiskAssessmentRequest(policy_document_id="missing-policy")
        )

    assert exc_info.value.status_code == 404
    assert str(exc_info.value) == "Policy not found or empty."
    service._fetch_v4_statute_docs.assert_not_awaited()
    service._llm_client.risk_assessment.assert_not_awaited()
