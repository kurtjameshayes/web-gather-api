"""Regression tests for compliance report section assembly.

Covers include-flag gating, field truncation, company fallback, and the
run_now path that chooses jurisdictions and still requests a health score
when that section is omitted from the markdown.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreResponse,
    MultiJurisdictionalResponse,
    ReportRequest,
    StrictestDenominatorItem,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _service(storage: MagicMock, v4: MagicMock | None = None) -> ComplianceSuiteService:
    return ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=load_config(),
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=storage,
        rate_limiter=MagicMock(),
        gap_analysis_v4_service=v4,
    )


def _stored_doc() -> dict:
    return {
        "policy_document_id": "pol-1",
        "company_name": None,
        "analyzed_at": "2026-04-01T00:00:00+00:00",
        "gaps": [
            {
                "jurisdiction": "CA",
                "statute_name": "S" * 25,
                "section": "E" * 35,
                "statute_chunk_id": "chunk-9",
                "requirement_summary": "R" * 70,
                "status": "partial",
                "policy_quote": "quote-ok",
                "conflict_description": "conflict-ok",
            }
        ],
        "summary": {
            "total_requirements": 9,
            "addressed": 1,
            "missing": 2,
            "conflicts": 3,
            "partial": 4,
            "ambiguous": 5,
        },
        "privacy_health_score": 90,
        "score_assessment": "Stored assessment stays verbatim.",
        "score_breakdown": {"by_jurisdiction": {"CA": 80, "VA": 0}},
        "strictest_common_denominator": [
            {
                "label": "Right To Know",
                "strictest_jurisdiction": "CA",
                "strictest_description": "D" * 250,
            }
        ],
    }


def test_report_markdown_keeps_selected_sections_and_cuts_cells() -> None:
    storage = MagicMock()
    storage.get_last_compliance_result = AsyncMock(return_value=_stored_doc())
    service = _service(storage)
    req = ReportRequest(
        policy_document_id="pol-1",
        format="markdown",
        source="latest_stored",
        include_gap=True,
        include_health_score=True,
        include_multi_jurisdictional=True,
    )

    payload = asyncio.run(service.report(req))

    content = payload["content"]
    assert payload["format"] == "markdown"
    assert content.startswith("# Compliance Report\n")
    assert "**Policy ID:** pol-1" in content
    assert "**Company:** —" in content
    assert "**Analyzed at:** 2026-04-01T00:00:00+00:00" in content
    assert "- Total requirements: 9" in content
    assert "- Addressed: 1" in content
    assert "- Missing: 2" in content
    assert "- Conflicts: 3" in content
    assert "- Partial:" not in content
    assert "- Ambiguous:" not in content
    assert (
        "| CA | "
        + ("S" * 20)
        + " | "
        + ("E" * 30)
        + " | chunk-9 | "
        + ("R" * 60)
        + " | partial | quote-ok | conflict-ok |"
    ) in content
    assert "S" * 21 not in content
    assert "E" * 31 not in content
    assert "R" * 61 not in content
    assert "**Score:** 90/100" in content
    assert "**Assessment:** Stored assessment stays verbatim." in content
    assert "**By jurisdiction:**" in content
    assert "- CA: 80" in content
    assert "- VA: 0" in content
    assert "## Strictest Common Denominator" in content
    assert "- **Right To Know** (strictest: CA)" in content
    assert "D" * 200 in content
    assert "D" * 201 not in content
    storage.get_last_compliance_result.assert_awaited_once_with("pol-1")


def test_report_flags_omit_sections_that_are_present_in_the_stored_doc() -> None:
    storage = MagicMock()
    storage.get_last_compliance_result = AsyncMock(return_value=_stored_doc())
    service = _service(storage)
    req = ReportRequest(
        policy_document_id="pol-1",
        format="markdown",
        source="latest_stored",
        include_gap=False,
        include_health_score=False,
        include_multi_jurisdictional=False,
    )

    content = asyncio.run(service.report(req))["content"]

    assert "**Policy ID:** pol-1" in content
    assert "## Gap Analysis" not in content
    assert "## Privacy Health Score" not in content
    assert "## Strictest Common Denominator" not in content
    assert "Stored assessment stays verbatim." not in content
    assert "quote-ok" not in content


def test_empty_gap_list_omits_the_gap_section() -> None:
    service = _service(MagicMock())
    content = service._build_report_markdown(
        {
            "policy_document_id": "pol-1",
            "company_name": "",
            "analyzed_at": "t",
            "gaps": [],
            "summary": {"total_requirements": 0},
        },
        include_gap=True,
        include_health_score=False,
        include_multi_jurisdictional=False,
    )
    assert "**Company:** —" in content
    assert "## Gap Analysis" not in content


def test_whitespace_company_name_is_kept() -> None:
    service = _service(MagicMock())
    content = service._build_report_markdown(
        {
            "policy_document_id": "pol-1",
            "company_name": "  ",
            "analyzed_at": "t",
            "gaps": [],
        },
        include_gap=False,
        include_health_score=False,
    )
    assert "**Company:**   " in content
    assert "—" not in content


def test_whitespace_policy_id_is_stripped_before_lookup() -> None:
    storage = MagicMock()
    storage.get_last_compliance_result = AsyncMock(return_value=None)
    service = _service(storage)

    with pytest.raises(ComplianceSuiteServiceError) as blank:
        asyncio.run(
            service.report(
                ReportRequest(
                    policy_document_id="   ",
                    format="markdown",
                    source="latest_stored",
                )
            )
        )
    assert blank.value.status_code == 400
    assert str(blank.value) == "policy_document_id is required."
    storage.get_last_compliance_result.assert_not_called()

    storage.get_last_compliance_result = AsyncMock(
        return_value={
            "policy_document_id": "pol-1",
            "company_name": "Acme",
            "analyzed_at": "t",
            "gaps": [],
        }
    )
    asyncio.run(
        service.report(
            ReportRequest(
                policy_document_id="  pol-1  ",
                format="markdown",
                source="latest_stored",
            )
        )
    )
    storage.get_last_compliance_result.assert_awaited_once_with("pol-1")


def _gap_response(jurisdictions: list[str]) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        company_name="Acme",
        applicable_jurisdictions=jurisdictions,
        analyzed_at="2026-04-02T00:00:00+00:00",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="§ 1",
                requirement_summary="Must disclose sale",
                status="missing",
            )
        ],
        summary=GapSummary(total_requirements=1, missing=1),
    )


def _health_response() -> HealthScoreResponse:
    return HealthScoreResponse(
        policy_document_id="pol-1",
        company_name="Acme",
        privacy_health_score=10,
        score_assessment="health-only-text",
        score_breakdown={"by_jurisdiction": {"CA": 10}},
        analyzed_at="2026-04-02T00:00:00+00:00",
    )


def test_run_now_requests_health_score_even_when_the_section_is_omitted() -> None:
    storage = MagicMock()
    storage.get_last_compliance_result = AsyncMock()
    v4 = MagicMock()
    v4.run = AsyncMock(return_value=_gap_response(["CA"]))
    service = _service(storage, v4=v4)
    service.health_score = AsyncMock(return_value=_health_response())
    service.multi_jurisdictional = AsyncMock()

    content = asyncio.run(
        service.report(
            ReportRequest(
                policy_document_id="pol-1",
                format="markdown",
                source="run_now",
                applicable_jurisdictions=["TX"],
                include_gap=True,
                include_health_score=False,
                include_multi_jurisdictional=False,
            )
        )
    )["content"]

    gap_req = v4.run.await_args.args[0]
    health_req = service.health_score.await_args.args[0]
    assert gap_req.save_results is False
    assert gap_req.policy_document_id == "pol-1"
    assert gap_req.applicable_jurisdictions == ["TX"]
    assert health_req.save_results is False
    assert health_req.policy_document_id == "pol-1"
    assert health_req.applicable_jurisdictions == ["TX"]
    service.multi_jurisdictional.assert_not_called()
    storage.get_last_compliance_result.assert_not_called()
    assert "Must disclose sale" in content
    assert "health-only-text" not in content
    assert "## Privacy Health Score" not in content


def test_run_now_multi_uses_request_jurisdictions_then_gap_jurisdictions() -> None:
    storage = MagicMock()
    v4 = MagicMock()
    v4.run = AsyncMock(return_value=_gap_response(["VA"]))
    service = _service(storage, v4=v4)
    service.health_score = AsyncMock(return_value=_health_response())
    service.multi_jurisdictional = AsyncMock(
        return_value=MultiJurisdictionalResponse(
            applicable_jurisdictions=["TX"],
            analyzed_at="2026-04-02T00:00:00+00:00",
            strictest_common_denominator=[
                StrictestDenominatorItem(
                    canonical_requirement_id="right_to_know",
                    label="Right To Know",
                    strictest_jurisdiction="TX",
                    strictest_description="sale opt-out",
                )
            ],
        )
    )

    explicit = asyncio.run(
        service.report(
            ReportRequest(
                policy_document_id="pol-1",
                format="markdown",
                source="run_now",
                applicable_jurisdictions=["TX"],
                include_health_score=False,
                include_multi_jurisdictional=True,
            )
        )
    )["content"]
    assert service.multi_jurisdictional.await_args.args[0].applicable_jurisdictions == ["TX"]
    assert "sale opt-out" in explicit

    v4.run = AsyncMock(return_value=_gap_response(["VA"]))
    service.multi_jurisdictional = AsyncMock(
        return_value=MultiJurisdictionalResponse(
            applicable_jurisdictions=["VA"],
            analyzed_at="2026-04-02T00:00:00+00:00",
            strictest_common_denominator=[
                StrictestDenominatorItem(
                    canonical_requirement_id="right_to_delete",
                    label="Right To Delete",
                    strictest_jurisdiction="VA",
                    strictest_description="delete window",
                )
            ],
        )
    )
    for jurisdictions in (None, []):
        asyncio.run(
            service.report(
                ReportRequest(
                    policy_document_id="pol-1",
                    format="markdown",
                    source="run_now",
                    applicable_jurisdictions=jurisdictions,
                    include_health_score=False,
                    include_multi_jurisdictional=True,
                )
            )
        )
        assert service.multi_jurisdictional.await_args.args[0].applicable_jurisdictions == ["VA"]


def test_run_now_without_v4_uses_gap_analysis_and_still_requests_health() -> None:
    storage = MagicMock()
    service = _service(storage, v4=None)
    service.gap_analysis = AsyncMock(return_value=_gap_response(["CA"]))
    service.health_score = AsyncMock(return_value=_health_response())
    service.multi_jurisdictional = AsyncMock()

    asyncio.run(
        service.report(
            ReportRequest(
                policy_document_id="pol-1",
                format="markdown",
                source="run_now",
                include_multi_jurisdictional=False,
            )
        )
    )

    gap_req = service.gap_analysis.await_args.args[0]
    health_req = service.health_score.await_args.args[0]
    assert service.gap_analysis.await_count == 1
    assert gap_req.save_results is False
    assert gap_req.policy_document_id == "pol-1"
    assert health_req.save_results is False
    service.multi_jurisdictional.assert_not_called()
