"""Focused regression tests for ComplianceSuiteService scoring behavior."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisResponse, GapItem, GapSummary, HealthScoreRequest
from compliance_suite_service import ComplianceSuiteService


def _build_service_with_gap_result(gap_result: GapAnalysisResponse) -> tuple[ComplianceSuiteService, AsyncMock]:
    config = load_config()
    storage = AsyncMock()
    rate_limiter = AsyncMock()
    rate_limiter.allow.return_value = True

    gap_v4_service = MagicMock()
    gap_v4_service.run = AsyncMock(return_value=gap_result)

    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=storage,
        rate_limiter=rate_limiter,
        gap_analysis_v4_service=gap_v4_service,
    )
    return service, storage


def test_health_score_applies_longest_weight_partial_credit_and_conflict_penalty() -> None:
    gap_result = GapAnalysisResponse(
        policy_document_id="policy-123",
        company_name="Acme",
        applicable_jurisdictions=["CA", "VA"],
        analyzed_at="2026-03-14T10:00:00Z",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="CCPA-1",
                requirement_summary="Consumer right to know what personal information is collected.",
                status="addressed",
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="CCPA-2",
                requirement_summary="Consumer right to delete personal information.",
                status="partial",
            ),
            GapItem(
                jurisdiction="VA",
                statute_reference="VCDPA-1",
                requirement_summary="Data retention limits must be disclosed.",
                status="missing",
            ),
            GapItem(
                jurisdiction="VA",
                statute_reference="VCDPA-2",
                requirement_summary="Right to correction should be clearly documented.",
                status="ambiguous",
            ),
            GapItem(
                jurisdiction="VA",
                statute_reference="VCDPA-3",
                requirement_summary="Right to access process details.",
                status="addressed",
                analysis_failed=True,
            ),
        ],
        summary=GapSummary(total_requirements=5, addressed=1, missing=1, partial=1, ambiguous=1, conflicts=1),
    )
    service, storage = _build_service_with_gap_result(gap_result)

    response = asyncio.run(
        service.health_score(
            HealthScoreRequest(
                policy_document_id="policy-123",
                save_results=False,
                weights={
                    "right to": 1.1,
                    "right to know": 2.4,
                    "data retention": 1.6,
                },
            )
        )
    )

    # Weighted ratio before conflict penalty is 2.95 / 6.2; conflict penalty (0.7) reduces score to 33.
    assert response.privacy_health_score == 33
    assert response.score_breakdown["by_jurisdiction"]["CA"] == 84
    assert response.score_breakdown["by_jurisdiction"]["VA"] == 0
    assert response.components["raw_ratio"] == 0.3331
    assert response.components["conflict_penalty_applied"] is True
    storage.write_compliance_result.assert_not_called()


def test_health_score_returns_no_applicable_statutes_when_gap_list_empty() -> None:
    gap_result = GapAnalysisResponse(
        policy_document_id="policy-empty",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-14T10:00:00Z",
        gaps=[],
        summary=GapSummary(total_requirements=0),
    )
    service, _ = _build_service_with_gap_result(gap_result)

    response = asyncio.run(
        service.health_score(
            HealthScoreRequest(policy_document_id="policy-empty", save_results=False)
        )
    )

    assert response.privacy_health_score is None
    assert response.error == "no_applicable_statutes"
    assert response.components["requirements_total"] == 0


def test_health_score_returns_insufficient_analysis_when_all_items_failed() -> None:
    gap_result = GapAnalysisResponse(
        policy_document_id="policy-failed",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-14T10:00:00Z",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="CCPA-1",
                requirement_summary="Right to know request intake.",
                status="missing",
                analysis_failed=True,
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="CCPA-2",
                requirement_summary="Right to delete request handling.",
                status="addressed",
                analysis_failed=True,
            ),
        ],
        summary=GapSummary(total_requirements=2, missing=1, addressed=1, analysis_failures=2),
    )
    service, _ = _build_service_with_gap_result(gap_result)

    response = asyncio.run(
        service.health_score(
            HealthScoreRequest(policy_document_id="policy-failed", save_results=False)
        )
    )

    assert response.privacy_health_score is None
    assert response.error == "insufficient_analysis"
    assert response.components["requirements_total"] == 2
