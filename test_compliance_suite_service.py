"""Regression tests for compliance suite health-score logic."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreRequest,
)
from compliance_suite_service import ComplianceSuiteService


def _build_service(gap_analysis_v4_service: object) -> ComplianceSuiteService:
    config = load_config()
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=rate_limiter,
        gap_analysis_v4_service=gap_analysis_v4_service,
    )


def _build_gap_response(gaps: list[GapItem]) -> GapAnalysisResponse:
    summary = GapSummary(
        total_requirements=len(gaps),
        addressed=sum(1 for g in gaps if g.status == "addressed"),
        missing=sum(1 for g in gaps if g.status == "missing"),
        conflicts=sum(1 for g in gaps if g.status == "conflict"),
        partial=sum(1 for g in gaps if g.status == "partial"),
        ambiguous=sum(1 for g in gaps if g.status == "ambiguous"),
    )
    return GapAnalysisResponse(
        policy_document_id="policy-1",
        company_name="Example Co",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-11T10:00:00+00:00",
        gaps=gaps,
        summary=summary,
    )


def test_health_score_uses_v4_service_and_default_substring_weights() -> None:
    """Default config weighting should apply phrase matches like 'right to know'."""
    gap_result = _build_gap_response(
        [
            GapItem(
                jurisdiction="CA",
                statute_reference="1798.100",
                requirement_summary="Consumers have a RIGHT TO KNOW what data is collected.",
                status="addressed",
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="1798.130",
                requirement_summary="Provide a privacy request intake channel.",
                status="missing",
            ),
        ]
    )
    gap_v4 = MagicMock()
    gap_v4.run = AsyncMock(return_value=gap_result)

    service = _build_service(gap_v4)
    service.gap_analysis = AsyncMock(side_effect=AssertionError("fallback gap_analysis should not run"))

    response = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="policy-1")))

    # Weighted score: 1.5 addressed / (1.5 + 1.0 total) = 0.6 -> 60
    assert response.privacy_health_score == 60
    gap_v4.run.assert_awaited_once()
    service.gap_analysis.assert_not_called()


def test_health_score_prefers_longest_weight_substring_match() -> None:
    """More specific phrase keys must win over broad overlapping keys."""
    gap_result = _build_gap_response(
        [
            GapItem(
                jurisdiction="CA",
                statute_reference="1798.105",
                requirement_summary="Users have a right to delete personal data.",
                status="addressed",
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="1798.121",
                requirement_summary="Users have a right to object to profiling.",
                status="missing",
            ),
        ]
    )
    gap_v4 = MagicMock()
    gap_v4.run = AsyncMock(return_value=gap_result)
    service = _build_service(gap_v4)

    request = HealthScoreRequest(
        policy_document_id="policy-1",
        weights={
            "right to": 1.0,
            "right to delete": 3.0,
        },
    )
    response = asyncio.run(service.health_score(request))

    # Longest match behavior: 3.0 addressed / (3.0 + 1.0 total) = 0.75 -> 75
    assert response.privacy_health_score == 75
    assert response.score_breakdown["by_jurisdiction"]["CA"] == 75
