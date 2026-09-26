"""Privacy health score weighting, conflict penalty, and assessment bands."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreRequest,
    _score_assessment,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _gap(
    jurisdiction: str,
    summary: str,
    status: str,
    *,
    analysis_failed: bool = False,
) -> GapItem:
    return GapItem(
        jurisdiction=jurisdiction,
        statute_reference="CCPA",
        requirement_summary=summary,
        status=status,
        analysis_failed=analysis_failed,
    )


def _gap_result(gaps, summary: GapSummary | None = None) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        company_name="Acme",
        applicable_jurisdictions=["CA", "VA", "CO"],
        analyzed_at="2026-03-21T00:00:00+00:00",
        gaps=gaps,
        summary=summary or GapSummary(),
    )


def _service(config, gap_result, *, allow: bool = True, v4_present: bool = True):
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=allow)
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    v4 = MagicMock()
    v4.run = AsyncMock(return_value=gap_result)
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=storage,
        rate_limiter=limiter,
        gap_analysis_v4_service=v4 if v4_present else None,
    )
    return service, v4, storage


def test_score_assessment_bands_conflicts_and_ambiguous_note() -> None:
    unable = _score_assessment(None, "no_applicable_statutes")
    assert unable.startswith("Unable to assess the privacy health score.")

    excellent = _score_assessment(
        90,
        None,
        {"requirements_total": 10, "addressed": 9, "conflicts": 1, "ambiguous": 2, "partial": 0, "missing": 0},
    )
    assert excellent.startswith("Excellent. Your policy fully addresses 9 of 10")
    assert "1 requirement(s)" in excellent
    assert "ambiguous" not in excellent

    good = _score_assessment(75, None, {"requirements_total": 4, "addressed": 3, "partial": 1})
    assert good.startswith("Good. Your policy addresses 3 of 4 requirements, with 1 partially addressed.")
    assert "ambiguous" not in good

    fair = _score_assessment(
        50,
        None,
        {"requirements_total": 4, "addressed": 2, "partial": 1, "missing": 1, "ambiguous": 2, "conflicts": 0},
    )
    assert fair.startswith("Fair.")
    assert "2 requirement(s) were marked ambiguous" in fair

    needs = _score_assessment(25, None, {"requirements_total": 4, "addressed": 1, "missing": 2, "partial": 1})
    assert needs.startswith("Needs improvement.")

    critical = _score_assessment(24, None, {"requirements_total": 4, "addressed": 0, "missing": 3})
    assert critical.startswith("Critical.")
    assert "3 missing" in critical

    no_counts = _score_assessment(90, None, {"requirements_total": 0})
    assert "strong alignment with applicable privacy statutes" in no_counts

    assert _score_assessment(89, None, {"requirements_total": 1, "addressed": 1}).startswith("Good.")
    assert _score_assessment(74, None, {"requirements_total": 1, "addressed": 0, "missing": 1}).startswith("Fair.")
    assert _score_assessment(49, None, {"requirements_total": 1, "addressed": 0, "missing": 1}).startswith(
        "Needs improvement."
    )
    assert _score_assessment(0, None, None).startswith("Critical.")


def test_health_score_weights_partial_credit_and_conflict_penalty() -> None:
    """Default weights, longest-key match, skipped failures, and summary-driven penalty."""
    config = load_config()
    assert config.partial_credit_percent == 0.5
    assert config.conflict_penalty_multiplier == 0.7

    gaps = [
        _gap("CA", "Consumers have a right to delete personal information", "addressed"),
        _gap("CA", "Limit data retention to one year", "partial"),
        _gap("VA", "Miscellaneous notice", "missing"),
        _gap("VA", "Consumers have a right to know", "conflict"),
        _gap("CO", "The processor must delete on request", "ambiguous"),
        _gap("CO", "right to delete", "addressed", analysis_failed=True),
    ]
    summary = GapSummary(
        total_requirements=6,
        addressed=2,
        missing=1,
        conflicts=1,
        partial=1,
        ambiguous=1,
        analysis_failures=1,
    )
    service, v4, storage = _service(config, _gap_result(gaps, summary))
    result = asyncio.run(
        service.health_score(
            HealthScoreRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA", "VA"],
                database="privacy-compliance",
                policy_collection="policies",
                save_results=False,
            )
        )
    )

    # Weights: delete 1.5, data retention 1.2 (not the shorter "retention" key),
    # unknown 1.0, right to know 1.5, processor must 1.0. Failed row is excluded.
    # weighted_sum = 1.5 + 0.5*1.2 = 2.1; total = 6.2; * 0.7 conflict penalty.
    assert result.privacy_health_score == 24
    assert result.components["raw_ratio"] == 0.2371
    assert result.components["conflict_penalty_applied"] is True
    assert result.components["addressed"] == 2
    assert result.components["requirements_total"] == 6
    assert result.score_assessment.startswith("Critical.")
    # Jurisdiction scores ignore the conflict penalty.
    assert result.score_breakdown["by_jurisdiction"] == {"CA": 78, "VA": 0, "CO": 0}
    assert result.error is None

    forwarded = v4.run.call_args[0][0]
    assert forwarded.policy_document_id == "pol-1"
    assert forwarded.applicable_jurisdictions == ["CA", "VA"]
    assert forwarded.database == "privacy-compliance"
    assert forwarded.policy_collection == "policies"
    assert forwarded.save_results is False
    storage.write_compliance_result.assert_not_called()


def test_weight_resolution_exact_key_beats_longer_substring() -> None:
    config = load_config()
    exact = "alpha key " + ("z" * 40)
    assert len(exact) == 50
    requirement = exact + " and right to delete"
    config.requirement_weights = {exact: 4.0, "right to delete": 5.0, "right to": 2.0}
    config.conflict_penalty_multiplier = 1.0

    gaps = [_gap("CA", requirement, "addressed")]
    service, _, storage = _service(
        config,
        _gap_result(gaps, GapSummary(total_requirements=1, addressed=1)),
    )
    result = asyncio.run(
        service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=True))
    )
    assert result.privacy_health_score == 100
    assert result.components["conflict_penalty_applied"] is False
    saved = storage.write_compliance_result.call_args[0][0]
    assert saved["privacy_health_score"] == 100
    assert saved["run_types"] == ["health_score"]
    assert saved["policy_document_id"] == "pol-1"


def test_longest_substring_match_is_case_insensitive() -> None:
    config = load_config()
    config.requirement_weights = {"right to": 2.0, "right to delete": 5.0, "delete": 9.0}
    config.conflict_penalty_multiplier = 0.7
    gaps = [_gap("CA", "Consumers Have The Right To Delete data", "addressed")]
    service, _, _ = _service(config, _gap_result(gaps, GapSummary(total_requirements=1, addressed=1, conflicts=0)))
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    # Longest contained key wins. Summary conflicts is 0, so the 0.7 multiplier is not applied.
    assert result.privacy_health_score == 100
    assert result.components["raw_ratio"] == 1.0
    assert result.components["conflict_penalty_applied"] is False


def test_summary_conflicts_apply_penalty_even_without_conflict_status() -> None:
    config = load_config()
    config.requirement_weights = {}
    config.conflict_penalty_multiplier = 0.5
    gaps = [_gap("CA", "Plain requirement", "addressed")]
    summary = GapSummary(total_requirements=1, addressed=1, conflicts=3)
    service, _, _ = _service(config, _gap_result(gaps, summary))
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    assert result.privacy_health_score == 50
    assert result.components["conflict_penalty_applied"] is True
    assert result.components["conflicts"] == 3


def test_penalty_skipped_when_multiplier_is_not_below_one() -> None:
    config = load_config()
    config.requirement_weights = {}
    config.conflict_penalty_multiplier = 1.0
    gaps = [_gap("CA", "Plain requirement", "missing"), _gap("VA", "Other duty", "addressed")]
    summary = GapSummary(total_requirements=2, addressed=1, missing=1, conflicts=4)
    service, _, _ = _service(config, _gap_result(gaps, summary))
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    assert result.privacy_health_score == 50
    assert result.components["conflict_penalty_applied"] is False
    assert result.score_breakdown["by_jurisdiction"] == {"CA": 0, "VA": 100}


def test_score_clamps_and_negative_weight_floors_at_zero() -> None:
    config = load_config()
    config.requirement_weights = {"bonus": 1.0}
    config.partial_credit_percent = 2.0
    config.conflict_penalty_multiplier = 1.0
    gaps = [_gap("CA", "bonus language", "partial")]
    service, _, _ = _service(config, _gap_result(gaps, GapSummary(total_requirements=1, partial=1)))
    high = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    assert high.privacy_health_score == 100
    assert high.components["raw_ratio"] == 2.0
    assert high.score_breakdown["by_jurisdiction"] == {"CA": 100}

    config.requirement_weights = {"penalty item": -1.0, "other duty": 3.0}
    config.partial_credit_percent = 0.5
    gaps = [
        _gap("CA", "penalty item", "addressed"),
        _gap("VA", "other duty", "missing"),
    ]
    service, _, _ = _service(config, _gap_result(gaps, GapSummary(total_requirements=2, addressed=1, missing=1)))
    low = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    assert low.privacy_health_score == 0
    # A negative jurisdiction denominator is omitted; it is not clamped to 0.
    assert "CA" not in low.score_breakdown["by_jurisdiction"]
    assert low.score_breakdown["by_jurisdiction"]["VA"] == 0


def test_empty_gaps_and_all_failed_rows_do_not_persist() -> None:
    config = load_config()
    service, v4, storage = _service(config, _gap_result([]))
    empty = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=True)))
    assert empty.privacy_health_score is None
    assert empty.error == "no_applicable_statutes"
    assert empty.score_assessment.startswith("Unable to assess")
    assert empty.components == {"requirements_total": 0, "addressed": 0, "missing": 0, "conflicts": 0}
    storage.write_compliance_result.assert_not_called()
    v4.run.assert_awaited()

    failed = [_gap("CA", "right to delete", "addressed", analysis_failed=True)]
    service, _, storage = _service(
        config,
        _gap_result(failed, GapSummary(total_requirements=1, analysis_failures=1)),
    )
    insufficient = asyncio.run(
        service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=True))
    )
    assert insufficient.privacy_health_score is None
    assert insufficient.error == "insufficient_analysis"
    assert insufficient.components["requirements_total"] == 1
    storage.write_compliance_result.assert_not_called()


def test_zero_weight_only_is_insufficient_analysis() -> None:
    config = load_config()
    config.requirement_weights = {"widget": 0.0}
    gaps = [_gap("CA", "widget duty", "addressed")]
    service, _, storage = _service(config, _gap_result(gaps, GapSummary(total_requirements=1, addressed=1)))
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))
    assert result.error == "insufficient_analysis"
    assert result.privacy_health_score is None
    storage.write_compliance_result.assert_not_called()


def test_rate_limit_and_v4_fallback() -> None:
    config = load_config()
    service, v4, _ = _service(config, _gap_result([]), allow=False)
    try:
        asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))
        raise AssertionError("expected rate limit")
    except ComplianceSuiteServiceError as exc:
        assert exc.status_code == 429
        assert "Rate limit" in str(exc)
    v4.run.assert_not_called()

    fallback_result = _gap_result([])
    service, _, _ = _service(config, fallback_result, v4_present=False)
    service.gap_analysis = AsyncMock(return_value=fallback_result)
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False)))
    assert result.error == "no_applicable_statutes"
    service.gap_analysis.assert_awaited()


def test_persist_failure_still_returns_the_score() -> None:
    config = load_config()
    config.requirement_weights = {}
    config.conflict_penalty_multiplier = 1.0
    gaps = [_gap("CA", "Plain requirement", "addressed")]
    service, _, storage = _service(config, _gap_result(gaps, GapSummary(total_requirements=1, addressed=1)))
    storage.write_compliance_result = AsyncMock(side_effect=RuntimeError("db down"))
    result = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=True)))
    assert result.privacy_health_score == 100
    assert result.company_name == "Acme"
