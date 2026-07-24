"""Regression tests for compliance report and multi-jurisdictional orchestration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreResponse,
    MultiJurisdictionalRequest,
    MultiJurisdictionalResponse,
    ReportRequest,
    StrictestDenominatorItem,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _service(**overrides: object) -> ComplianceSuiteService:
    config = load_config()
    config.canonical_requirement_ids = ["right_to_know", "right_to_delete"]
    config.default_jurisdictions = ["CA", "VA"]
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)
    llm_client = MagicMock()
    llm_client.requirement_extraction = AsyncMock(return_value={"requirements": []})
    llm_client.strictness_comparison = AsyncMock(return_value={})
    kwargs = {
        "mongo_client": MagicMock(),
        "config": config,
        "retriever": MagicMock(),
        "llm_client": llm_client,
        "storage": MagicMock(),
        "rate_limiter": rate_limiter,
        "gap_analysis_v4_service": None,
    }
    kwargs.update(overrides)
    return ComplianceSuiteService(**kwargs)


# ----- multi_jurisdictional -----


@pytest.mark.anyio
async def test_multi_jurisdictional_extracts_and_compares_requirements() -> None:
    """Requirement extraction feeds canonical matching and strictness comparison."""
    service = _service()
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {
                    "header_text": "Cal. Civ. Code § 1798.100",
                    "subtopic_text": "Consumers have the right to know what is collected.",
                },
                {"requirement_summary": ""},  # skipped: empty text
            ],
            "VA": [
                {
                    "header_text": "Va. Code § 59.1-577",
                    "subtopic_text": "Consumers may request deletion of personal data.",
                }
            ],
        }
    )
    service._llm_client.requirement_extraction = AsyncMock(
        side_effect=[
            {
                "requirements": [
                    {
                        "label": "Right to know",
                        "description": "Disclose categories of personal information collected.",
                    }
                ]
            },
            {
                "requirements": [
                    {
                        "label": "Right to delete",
                        "description": "Delete personal data upon verified request.",
                    }
                ]
            },
        ]
    )
    service._llm_client.strictness_comparison = AsyncMock(
        side_effect=[
            {
                "strictest_jurisdiction": "CA",
                "strictest_description": "Disclose categories collected and sold.",
            },
            {
                "strictest_jurisdiction": "VA",
                "strictest_description": "Delete personal data within 45 days.",
            },
        ]
    )

    result = await service.multi_jurisdictional(
        MultiJurisdictionalRequest(applicable_jurisdictions=["CA", "VA"])
    )

    service._fetch_v4_statute_docs.assert_awaited_once_with(["CA", "VA"])
    assert service._llm_client.requirement_extraction.await_count == 2
    service._llm_client.requirement_extraction.assert_any_await(
        "Cal. Civ. Code § 1798.100\n\nConsumers have the right to know what is collected."
    )
    service._llm_client.requirement_extraction.assert_any_await(
        "Va. Code § 59.1-577\n\nConsumers may request deletion of personal data."
    )

    assert result.applicable_jurisdictions == ["CA", "VA"]
    assert len(result.strictest_common_denominator) == 2
    know, delete = result.strictest_common_denominator
    assert know.canonical_requirement_id == "right_to_know"
    assert know.strictest_jurisdiction == "CA"
    assert know.strictest_description == "Disclose categories collected and sold."
    assert know.all_jurisdictions == ["CA", "VA"]
    assert know.policy_alignment == "not_provided"
    assert delete.canonical_requirement_id == "right_to_delete"
    assert delete.strictest_jurisdiction == "VA"
    assert result.conflicts_between_jurisdictions == []

    first_compare = service._llm_client.strictness_comparison.await_args_list[0]
    assert first_compare.args[0] == "right_to_know"
    assert first_compare.args[1] == [
        {
            "jurisdiction": "CA",
            "description": "Disclose categories of personal information collected.",
        }
    ]


@pytest.mark.anyio
async def test_multi_jurisdictional_falls_back_when_strictness_fields_missing() -> None:
    """Missing LLM strictness fields fall back to the first matched jurisdiction."""
    service = _service()
    service._config.canonical_requirement_ids = ["right_to_know"]
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [{"requirement_summary": "Right to know personal data categories."}],
            "CO": [{"requirement_summary": "Right to know categories of personal data."}],
        }
    )
    service._llm_client.requirement_extraction = AsyncMock(
        side_effect=[
            {
                "requirements": [
                    {"label": "Right to know", "description": "CA disclosure duty"}
                ]
            },
            {
                "requirements": [
                    {"label": "Right to know", "description": "CO disclosure duty"}
                ]
            },
        ]
    )
    service._llm_client.strictness_comparison = AsyncMock(return_value={})

    result = await service.multi_jurisdictional(
        MultiJurisdictionalRequest(applicable_jurisdictions=["CA", "CO"])
    )

    assert len(result.strictest_common_denominator) == 1
    item = result.strictest_common_denominator[0]
    assert item.strictest_jurisdiction == "CA"
    assert item.strictest_description == ""


@pytest.mark.anyio
async def test_multi_jurisdictional_rate_limit_short_circuits_work() -> None:
    """Rate limiting stops before statute retrieval or LLM calls."""
    service = _service()
    service._rate_limiter.allow = AsyncMock(return_value=False)
    service._fetch_v4_statute_docs = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.multi_jurisdictional(
            MultiJurisdictionalRequest(applicable_jurisdictions=["CA"])
        )

    assert exc_info.value.status_code == 429
    assert str(exc_info.value) == "Rate limit exceeded"
    service._fetch_v4_statute_docs.assert_not_awaited()
    service._llm_client.requirement_extraction.assert_not_awaited()
    service._llm_client.strictness_comparison.assert_not_awaited()


# ----- report / _build_report_markdown -----


def test_build_report_markdown_respects_include_flags_and_sanitizes_pipes() -> None:
    """Markdown sections honor include flags and escape pipe characters in table cells."""
    service = _service()
    markdown = service._build_report_markdown(
        {
            "policy_document_id": "policy-1",
            "company_name": "Acme",
            "analyzed_at": "2026-07-24T10:00:00Z",
            "summary": {
                "total_requirements": 2,
                "addressed": 1,
                "missing": 1,
                "conflicts": 0,
            },
            "gaps": [
                {
                    "jurisdiction": "CA",
                    "statute_name": "CCPA",
                    "section": "1798.100",
                    "statute_chunk_id": "chunk-1",
                    "requirement_summary": "Disclose collected categories",
                    "status": "missing",
                    "policy_quote": "We share | sell data",
                    "conflict_description": "A | B conflict",
                }
            ],
            "privacy_health_score": 72,
            "score_assessment": "Fair with improvement needed.",
            "score_breakdown": {"by_jurisdiction": {"CA": 70, "VA": 74}},
            "strictest_common_denominator": [
                {
                    "label": "Right To Know",
                    "strictest_jurisdiction": "CA",
                    "strictest_description": "Disclose categories collected.",
                }
            ],
        },
        include_gap=True,
        include_health_score=True,
        include_multi_jurisdictional=True,
    )

    assert "# Compliance Report" in markdown
    assert "**Policy ID:** policy-1" in markdown
    assert "**Company:** Acme" in markdown
    assert "## Gap Analysis" in markdown
    assert "- Missing: 1" in markdown
    assert "We share   sell data" in markdown
    assert "A   B conflict" in markdown
    assert "## Privacy Health Score" in markdown
    assert "**Score:** 72/100" in markdown
    assert "**Assessment:** Fair with improvement needed." in markdown
    assert "- CA: 70" in markdown
    assert "## Strictest Common Denominator" in markdown
    assert "**Right To Know** (strictest: CA)" in markdown

    gap_only = service._build_report_markdown(
        {
            "policy_document_id": "policy-1",
            "gaps": [{"jurisdiction": "CA", "requirement_summary": "x", "status": "missing"}],
            "privacy_health_score": 50,
            "strictest_common_denominator": [{"label": "Right To Know"}],
        },
        include_gap=True,
        include_health_score=False,
        include_multi_jurisdictional=False,
    )
    assert "## Gap Analysis" in gap_only
    assert "## Privacy Health Score" not in gap_only
    assert "## Strictest Common Denominator" not in gap_only


@pytest.mark.anyio
async def test_report_latest_stored_returns_markdown_from_storage() -> None:
    """latest_stored loads the persisted result and returns markdown content."""
    service = _service()
    service._storage.get_last_compliance_result = AsyncMock(
        return_value={
            "policy_document_id": "policy-1",
            "company_name": "Acme",
            "analyzed_at": "2026-07-24T10:00:00Z",
            "summary": {"total_requirements": 1, "addressed": 0, "missing": 1, "conflicts": 0},
            "gaps": [
                {
                    "jurisdiction": "CA",
                    "requirement_summary": "Disclose categories",
                    "status": "missing",
                }
            ],
            "privacy_health_score": 40,
            "score_assessment": "Needs work.",
        }
    )

    result = await service.report(
        ReportRequest(
            policy_document_id="policy-1",
            format="markdown",
            source="latest_stored",
            include_multi_jurisdictional=False,
        )
    )

    service._storage.get_last_compliance_result.assert_awaited_once_with("policy-1")
    assert result["format"] == "markdown"
    assert "## Gap Analysis" in result["content"]
    assert "**Score:** 40/100" in result["content"]


@pytest.mark.anyio
async def test_report_latest_stored_missing_result_is_404() -> None:
    """Missing stored compliance results fail before markdown generation."""
    service = _service()
    service._storage.get_last_compliance_result = AsyncMock(return_value=None)

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.report(
            ReportRequest(
                policy_document_id="missing-policy",
                format="markdown",
                source="latest_stored",
            )
        )

    assert exc_info.value.status_code == 404
    assert str(exc_info.value) == "No stored result for this policy."


@pytest.mark.anyio
async def test_report_requires_policy_document_id() -> None:
    """Blank policy IDs are rejected before storage or analysis work."""
    service = _service()
    service._storage.get_last_compliance_result = AsyncMock()

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.report(
            ReportRequest(
                policy_document_id="   ",
                format="markdown",
                source="latest_stored",
            )
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value) == "policy_document_id is required."
    service._storage.get_last_compliance_result.assert_not_awaited()


@pytest.mark.anyio
async def test_report_pdf_format_returns_501() -> None:
    """PDF requests are explicitly unsupported after markdown content is built."""
    service = _service()
    service._storage.get_last_compliance_result = AsyncMock(
        return_value={
            "policy_document_id": "policy-1",
            "gaps": [],
            "privacy_health_score": 90,
        }
    )

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.report(
            ReportRequest(
                policy_document_id="policy-1",
                format="pdf",
                source="latest_stored",
            )
        )

    assert exc_info.value.status_code == 501
    assert "PDF generation not implemented" in str(exc_info.value)


@pytest.mark.anyio
async def test_report_run_now_orchestrates_gap_health_and_optional_multi() -> None:
    """run_now combines live analyses without persisting intermediate results."""
    gap_service = MagicMock()
    gap_service.run = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="policy-1",
            company_name="Acme",
            applicable_jurisdictions=["CA", "VA"],
            analyzed_at="2026-07-24T12:00:00Z",
            gaps=[
                GapItem(
                    jurisdiction="CA",
                    statute_reference="§ 1798.100",
                    requirement_summary="Disclose categories",
                    status="missing",
                )
            ],
            summary=GapSummary(total_requirements=1, missing=1),
        )
    )
    service = _service(gap_analysis_v4_service=gap_service)
    service.health_score = AsyncMock(
        return_value=HealthScoreResponse(
            policy_document_id="policy-1",
            privacy_health_score=55,
            score_assessment="Fair.",
            score_breakdown={"by_jurisdiction": {"CA": 55}},
            components={"missing": 1, "addressed": 0},
            analyzed_at="2026-07-24T12:00:00Z",
        )
    )
    service.multi_jurisdictional = AsyncMock(
        return_value=MultiJurisdictionalResponse(
            applicable_jurisdictions=["CA", "VA"],
            analyzed_at="2026-07-24T12:00:00Z",
            strictest_common_denominator=[
                StrictestDenominatorItem(
                    canonical_requirement_id="right_to_know",
                    label="Right To Know",
                    strictest_jurisdiction="CA",
                    strictest_description="Disclose categories collected.",
                    all_jurisdictions=["CA", "VA"],
                )
            ],
        )
    )

    result = await service.report(
        ReportRequest(
            policy_document_id="policy-1",
            format="markdown",
            source="run_now",
            applicable_jurisdictions=["CA", "VA"],
            include_gap=True,
            include_health_score=True,
            include_multi_jurisdictional=True,
        )
    )

    gap_req = gap_service.run.await_args.args[0]
    assert gap_req.policy_document_id == "policy-1"
    assert gap_req.applicable_jurisdictions == ["CA", "VA"]
    assert gap_req.save_results is False

    health_req = service.health_score.await_args.args[0]
    assert health_req.policy_document_id == "policy-1"
    assert health_req.save_results is False

    multi_req = service.multi_jurisdictional.await_args.args[0]
    assert multi_req.applicable_jurisdictions == ["CA", "VA"]

    assert result["format"] == "markdown"
    assert "## Gap Analysis" in result["content"]
    assert "**Score:** 55/100" in result["content"]
    assert "## Strictest Common Denominator" in result["content"]
    assert "Right To Know" in result["content"]
