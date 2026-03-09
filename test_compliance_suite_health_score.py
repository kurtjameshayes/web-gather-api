"""Regression tests for compliance health-score behavior."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisResponse, GapItem, GapSummary, HealthScoreRequest
from compliance_suite_service import ComplianceSuiteService


def test_health_score_uses_v4_service_and_longest_substring_weight_match() -> None:
    config = load_config()
    config.conflict_penalty_multiplier = 1.0

    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()

    gap_result = GapAnalysisResponse(
        policy_document_id="policy-123",
        company_name="Acme",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-09T10:00:00+00:00",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="stat-1",
                requirement_summary="right to delete personal data on request",
                status="addressed",
                analysis_failed=False,
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="stat-2",
                requirement_summary="right to know categories of personal information collected",
                status="missing",
                analysis_failed=False,
            ),
        ],
        summary=GapSummary(total_requirements=2, missing=1, addressed=1, conflicts=0),
    )

    v4_service = MagicMock()
    v4_service.run = AsyncMock(return_value=gap_result)

    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=storage,
        rate_limiter=rate_limiter,
        gap_analysis_v4_service=v4_service,
    )
    service.gap_analysis = AsyncMock(side_effect=AssertionError("fallback gap_analysis should not be used"))

    req = HealthScoreRequest(
        policy_document_id="policy-123",
        save_results=False,
        weights={
            "right to": 0.2,
            "right to delete": 2.0,
        },
    )
    response = asyncio.run(service.health_score(req))

    assert response.privacy_health_score == 91
    assert response.score_breakdown["by_jurisdiction"]["CA"] == 91
    assert response.components["requirements_total"] == 2
    assert response.components["addressed"] == 1
    v4_service.run.assert_awaited_once()
    service.gap_analysis.assert_not_called()
