"""Service-level tests for ComplianceSuiteService.risk_assessment.

Route tests mock this method. These pin statute-summary limits, jurisdiction
fallback, report markdown, and the fact that the assessment is not persisted.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import RiskAssessmentRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


class _Allow:
    async def allow(self) -> bool:
        return True


class _Deny:
    async def allow(self) -> bool:
        return False


def _service(limiter: object) -> ComplianceSuiteService:
    config = load_config()
    config.default_jurisdictions = ["CA", "VA"]
    llm = MagicMock()
    llm.risk_assessment = AsyncMock(
        return_value={
            "processing_purposes": ["billing"],
            "data_categories": ["email"],
            "risks": [],
            "mitigations": [],
            "gaps_from_statute": [],
        }
    )
    storage = MagicMock()
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=storage,
        rate_limiter=limiter,
    )
    service._load_policy_text = AsyncMock(return_value=("policy body", "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(return_value={})
    return service


def test_risk_assessment_rate_limit_before_policy_load() -> None:
    service = _service(_Deny())

    async def run() -> None:
        with pytest.raises(ComplianceSuiteServiceError) as exc:
            await service.risk_assessment(
                RiskAssessmentRequest(policy_document_id="pol-1")
            )
        assert exc.value.status_code == 429
        assert str(exc.value) == "Rate limit exceeded"

    asyncio.run(run())
    service._load_policy_text.assert_not_called()
    service._llm_client.risk_assessment.assert_not_called()


def test_risk_assessment_empty_policy_is_404() -> None:
    service = _service(_Allow())
    service._load_policy_text = AsyncMock(return_value=("", None))

    async def run() -> None:
        with pytest.raises(ComplianceSuiteServiceError) as exc:
            await service.risk_assessment(
                RiskAssessmentRequest(policy_document_id="pol-1", applicable_jurisdictions=["CA"])
            )
        assert exc.value.status_code == 404
        assert str(exc.value) == "Policy not found or empty."

    asyncio.run(run())
    service._fetch_v4_statute_docs.assert_not_called()
    service._llm_client.risk_assessment.assert_not_called()
    service._storage.write_compliance_result.assert_not_called()


def test_risk_assessment_jurisdiction_fallback_and_template_id() -> None:
    service = _service(_Allow())

    async def run() -> None:
        omitted = await service.risk_assessment(
            RiskAssessmentRequest(policy_document_id="pol-1")
        )
        empty = await service.risk_assessment(
            RiskAssessmentRequest(policy_document_id="pol-1", applicable_jurisdictions=[])
        )
        explicit = await service.risk_assessment(
            RiskAssessmentRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["TX"],
                template_id="",
            )
        )
        padded = await service.risk_assessment(
            RiskAssessmentRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["TX"],
                template_id="  ",
            )
        )
        return omitted, empty, explicit, padded

    omitted, empty, explicit, padded = asyncio.run(run())
    assert omitted.applicable_jurisdictions == ["CA", "VA"]
    assert empty.applicable_jurisdictions == ["CA", "VA"]
    assert explicit.applicable_jurisdictions == ["TX"]
    assert explicit.template_id == "default"
    assert padded.template_id == "  "
    assert omitted.company_name == "Acme"
    assert omitted.report is None
    assert service._fetch_v4_statute_docs.await_args_list[0].args == (["CA", "VA"],)
    assert service._fetch_v4_statute_docs.await_args_list[1].args == (["CA", "VA"],)
    assert service._fetch_v4_statute_docs.await_args_list[2].args == (["TX"],)
    service._storage.write_compliance_result.assert_not_called()


def test_risk_assessment_statute_summary_limits() -> None:
    """Each statute chunk is cut at 500 characters, the first 5 docs count, then the joined summary is cut at 6000.

    A blank document still occupies one of the five slots. The service passes the full policy text through.
    """
    service = _service(_Allow())
    policy_text = "P" * 9000
    service._load_policy_text = AsyncMock(return_value=(policy_text, "Acme"))
    service._fetch_v4_statute_docs = AsyncMock(
        return_value={
            "CA": [
                {"subtopic_text": ""},
                {"subtopic_text": "A" * 501},
                {"subtopic_text": "two"},
                {"subtopic_text": "three"},
                {"subtopic_text": "four"},
                {"subtopic_text": "SHOULD_NOT_APPEAR"},
            ],
            "VA": [{"subtopic_text": "V" * 500} for _ in range(5)],
            "CO": [{"subtopic_text": "C" * 500} for _ in range(5)],
            "NY": [{"subtopic_text": "N" * 500}],
        }
    )

    async def run() -> None:
        return await service.risk_assessment(
            RiskAssessmentRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA", "VA", "CO", "NY"],
            )
        )

    asyncio.run(run())
    policy_arg, summary = service._llm_client.risk_assessment.await_args.args
    ca_parts = [
        "[CA] " + ("A" * 500),
        "[CA] two",
        "[CA] three",
        "[CA] four",
    ]
    later_parts = (
        [f"[VA] {'V' * 500}"] * 5
        + [f"[CO] {'C' * 500}"] * 5
        + ["[NY] " + ("N" * 500)]
    )
    expected = "\n\n".join(ca_parts + later_parts)[:6000]
    assert policy_arg == policy_text
    assert summary == expected
    assert len(summary) == 6000
    assert "A" * 501 not in summary
    assert "SHOULD_NOT_APPEAR" not in summary
    assert summary.startswith(ca_parts[0])
    assert "[CO] " in summary
    assert summary.endswith("N" * 385)
    assert "N" * 386 not in summary


def test_risk_assessment_report_renders_mixed_risks() -> None:
    service = _service(_Allow())
    service._llm_client.risk_assessment = AsyncMock(
        return_value={
            "processing_purposes": ["billing", ""],
            "data_categories": ["email"],
            "risks": [
                {"description": "over-collection", "severity": "high"},
                "unstructured risk",
                {"severity": "low"},
            ],
            "mitigations": ["minimize"],
            "gaps_from_statute": ["missing retention limit"],
        }
    )

    async def run() -> None:
        return await service.risk_assessment(
            RiskAssessmentRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                template_id="dpia",
                include_report=True,
            )
        )

    response = asyncio.run(run())
    assert response.template_id == "dpia"
    assert response.assessment["gaps_from_statute"] == ["missing retention limit"]
    assert response.report == (
        "# Risk Assessment\n"
        "**Policy:** pol-1\n"
        "\n"
        "## Processing purposes\n"
        "- billing\n"
        "- \n"
        "\n"
        "## Data categories\n"
        "- email\n"
        "\n"
        "## Risks\n"
        "- over-collection (severity: high)\n"
        "- unstructured risk\n"
        "-  (severity: low)\n"
        "\n"
        "## Mitigations\n"
        "- minimize"
    )
    service._storage.write_compliance_result.assert_not_called()
