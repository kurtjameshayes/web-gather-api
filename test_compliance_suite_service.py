"""Regression tests for compliance suite health-score behavior."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _build_service(gap_v4_service: object | None = None) -> ComplianceSuiteService:
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=load_config(),
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=SimpleNamespace(allow=AsyncMock(return_value=True)),
        gap_analysis_v4_service=gap_v4_service,
    )


def _gap_item(requirement_summary: str, status: str, jurisdiction: str = "CA") -> GapItem:
    return GapItem(
        jurisdiction=jurisdiction,
        statute_reference="statute-1",
        requirement_summary=requirement_summary,
        status=status,
    )


def _gap_response(gaps: list[GapItem]) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="policy-1",
        company_name="Acme",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-08T00:00:00Z",
        gaps=gaps,
        summary=GapSummary(
            total_requirements=len(gaps),
            addressed=sum(1 for g in gaps if g.status == "addressed"),
            missing=sum(1 for g in gaps if g.status == "missing"),
            conflicts=sum(1 for g in gaps if g.status == "conflict"),
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_health_score_uses_v4_gap_service_when_available(anyio_backend: str) -> None:
    del anyio_backend
    gap_result = _gap_response([
        _gap_item("Consumers have a right to delete personal data.", "addressed")
    ])
    gap_v4_service = SimpleNamespace(run=AsyncMock(return_value=gap_result))

    service = _build_service(gap_v4_service=gap_v4_service)
    service.gap_analysis = AsyncMock(side_effect=AssertionError("fallback gap_analysis should not be called"))

    response = await service.health_score(
        HealthScoreRequest(
            policy_document_id="policy-1",
            applicable_jurisdictions=["CA"],
            save_results=False,
        )
    )

    assert response.privacy_health_score == 100
    gap_v4_service.run.assert_awaited_once()
    called_req = gap_v4_service.run.await_args.args[0]
    assert called_req.policy_document_id == "policy-1"
    assert called_req.applicable_jurisdictions == ["CA"]
    assert called_req.save_results is False


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_health_score_prefers_longest_matching_weight_key(anyio_backend: str) -> None:
    del anyio_backend
    service = _build_service()
    service.gap_analysis = AsyncMock(
        return_value=_gap_response(
            [
                _gap_item("Consumers have a Right To Delete personal data.", "addressed"),
                _gap_item("Consumers retain a right to access disclosures.", "missing"),
            ]
        )
    )

    response = await service.health_score(
        HealthScoreRequest(
            policy_document_id="policy-1",
            weights={"right to": 1.1, "right to delete": 1.5},
            save_results=False,
        )
    )

    assert response.privacy_health_score == 58
    assert response.components["raw_ratio"] == 0.5769


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_health_score_exact_prefix_weight_beats_substring(anyio_backend: str) -> None:
    del anyio_backend
    service = _build_service()
    prefix = "X" * 50
    service.gap_analysis = AsyncMock(
        return_value=_gap_response(
            [
                _gap_item(f"{prefix} right to delete personal data", "addressed"),
                _gap_item("Generic requirement without configured weight.", "missing"),
            ]
        )
    )

    response = await service.health_score(
        HealthScoreRequest(
            policy_document_id="policy-1",
            weights={prefix: 0.2, "right to delete": 2.0},
            save_results=False,
        )
    )

    assert response.privacy_health_score == 17
    assert response.components["raw_ratio"] == 0.1667
