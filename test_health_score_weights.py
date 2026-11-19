"""Health-score weighting, partial credit, and conflict penalty."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreRequest,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _gap(
    jurisdiction: str,
    requirement_summary: str,
    status: str,
    *,
    analysis_failed: bool = False,
) -> GapItem:
    return GapItem(
        jurisdiction=jurisdiction,
        statute_reference="ref",
        requirement_summary=requirement_summary,
        status=status,
        analysis_failed=analysis_failed,
    )


def _gap_response(
    gaps: List[GapItem],
    summary: GapSummary,
    *,
    company_name: str = "Acme",
    analyzed_at: str = "2026-03-21T00:00:00+00:00",
) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        company_name=company_name,
        applicable_jurisdictions=["CA"],
        analyzed_at=analyzed_at,
        gaps=gaps,
        summary=summary,
    )


class _Limiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls = 0

    async def allow(self) -> bool:
        self.calls += 1
        return self.allowed


class _V4:
    def __init__(self, response: Optional[GapAnalysisResponse] = None) -> None:
        self.response = response
        self.requests: List[Any] = []

    async def run(self, req: Any) -> GapAnalysisResponse:
        self.requests.append(req)
        assert self.response is not None
        return self.response


class _Storage:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.docs: List[Dict[str, Any]] = []

    async def write_compliance_result(self, doc: Dict[str, Any]) -> str:
        if self.fail:
            raise RuntimeError("db down")
        self.docs.append(doc)
        return "stored"


def _service(
    *,
    partial_credit: float = 0.5,
    conflict_penalty: float = 0.7,
    config_weights: Optional[Dict[str, float]] = None,
    allowed: bool = True,
    fail_write: bool = False,
    gap_response: Optional[GapAnalysisResponse] = None,
) -> tuple[ComplianceSuiteService, _Limiter, _V4, _Storage]:
    config = load_config()
    config.partial_credit_percent = partial_credit
    config.conflict_penalty_multiplier = conflict_penalty
    if config_weights is not None:
        config.requirement_weights = config_weights
    limiter = _Limiter(allowed)
    v4 = _V4(gap_response)
    storage = _Storage(fail_write)
    service = ComplianceSuiteService(
        mongo_client=object(),
        config=config,
        retriever=object(),
        llm_client=object(),
        storage=storage,  # type: ignore[arg-type]
        rate_limiter=limiter,  # type: ignore[arg-type]
        gap_analysis_v4_service=v4,
    )
    return service, limiter, v4, storage


def test_exact_weight_beats_longer_substring_and_penalty_skips_jurisdictions() -> None:
    """Request weights replace config weights. Exact 50-char keys win before substring search.

    Partial credit is half the weight. Failed analyses are left out of both totals.
    A conflict penalty scales the overall ratio only; jurisdiction scores stay unpenalized.
    """
    long_key = "PREFIX-" + ("z" * 50)
    summary_exact = long_key + " tail"
    exact_key = summary_exact[:50]
    assert len(exact_key) == 50
    assert len(long_key) > 50

    weights = {
        exact_key: 4.0,
        long_key: 9.0,
        "right to delete": 1.5,
        "delete": 6.0,
        "right": 7.0,
    }
    gaps = [
        _gap("CA", summary_exact, "addressed"),
        _gap("CA", "Please honor the right to delete customer records", "partial"),
        _gap("NY", "Backup copies are retained for operations", "missing"),
        _gap("NY", "Vendor contracts may override this notice", "conflict"),
        _gap("TX", "Language is unclear about retention", "ambiguous"),
        _gap("EU", exact_key, "addressed", analysis_failed=True),
    ]
    summary = GapSummary(
        total_requirements=6,
        missing=1,
        addressed=1,
        conflicts=1,
        partial=1,
        ambiguous=1,
        analysis_failures=1,
    )
    service, limiter, v4, storage = _service(
        config_weights={"from-config": 99.0},
        gap_response=_gap_response(gaps, summary),
    )

    response = asyncio.run(
        service.health_score(
            HealthScoreRequest(policy_document_id="pol-1", weights=weights, save_results=True)
        )
    )

    assert limiter.calls == 1
    assert len(v4.requests) == 1
    assert v4.requests[0].policy_document_id == "pol-1"
    assert v4.requests[0].save_results is True
    assert response.privacy_health_score == 39
    assert response.error is None
    assert response.company_name == "Acme"
    assert response.analyzed_at == "2026-03-21T00:00:00+00:00"
    assert response.components["requirements_total"] == 6
    assert response.components["addressed"] == 1
    assert response.components["missing"] == 1
    assert response.components["conflicts"] == 1
    assert response.components["partial"] == 1
    assert response.components["ambiguous"] == 1
    assert response.components["raw_ratio"] == 0.3912
    assert response.components["conflict_penalty_applied"] is True
    assert response.score_breakdown == {"by_jurisdiction": {"CA": 86, "NY": 0, "TX": 0}, "by_category": {}}
    assert response.score_assessment.startswith("Needs improvement.")
    assert "addresses 1 of 6" in response.score_assessment
    assert "1 missing" in response.score_assessment
    assert "1 partially addressed" in response.score_assessment
    assert "1 requirement(s) where policy language conflicts" in response.score_assessment
    assert "1 requirement(s) were marked ambiguous" in response.score_assessment
    assert len(storage.docs) == 1
    stored = storage.docs[0]
    assert stored["policy_document_id"] == "pol-1"
    assert stored["company_name"] == "Acme"
    assert stored["privacy_health_score"] == 39
    assert stored["run_types"] == ["health_score"]
    assert stored["score_breakdown"]["by_jurisdiction"]["CA"] == 86
    assert "gaps" not in stored


def test_empty_request_weights_fall_through_to_config() -> None:
    """weights={} is falsy, so configured requirement weights still apply."""
    gaps = [
        _gap("CA", "the right to delete data", "addressed"),
        _gap("CA", "other duty", "missing"),
    ]
    summary = GapSummary(total_requirements=2, addressed=1, missing=1)
    service, _, _, storage = _service(
        conflict_penalty=0.7,
        config_weights={"right to delete": 2.0},
        gap_response=_gap_response(gaps, summary, company_name=None),
    )

    response = asyncio.run(
        service.health_score(HealthScoreRequest(policy_document_id="pol-1", weights={}))
    )

    assert response.privacy_health_score == 67
    assert response.components["raw_ratio"] == 0.6667
    assert response.components["conflict_penalty_applied"] is False
    assert response.score_assessment.startswith("Fair.")
    assert response.score_breakdown["by_jurisdiction"] == {"CA": 67}
    assert storage.docs[0]["privacy_health_score"] == 67


def test_request_weights_replace_config_weights() -> None:
    """A non-empty request weight map is not merged with requirement_weights."""
    gaps = [
        _gap("CA", "general notice", "addressed"),
        _gap("CA", "right to delete now", "missing"),
    ]
    summary = GapSummary(total_requirements=2, addressed=1, missing=1)
    service, _, _, _ = _service(
        config_weights={"right to delete": 8.0},
        gap_response=_gap_response(gaps, summary),
    )

    response = asyncio.run(
        service.health_score(
            HealthScoreRequest(
                policy_document_id="pol-1",
                weights={"unrelated phrase": 3.0},
            )
        )
    )

    assert response.privacy_health_score == 50
    assert response.components["raw_ratio"] == 0.5
    assert response.score_breakdown["by_jurisdiction"] == {"CA": 50}


def test_conflict_multiplier_of_one_does_not_reduce_score() -> None:
    gaps = [
        _gap("CA", "general notice", "addressed"),
        _gap("NY", "conflicting retention language", "conflict"),
    ]
    summary = GapSummary(total_requirements=2, addressed=1, conflicts=1)
    service, _, _, _ = _service(
        conflict_penalty=1.0,
        config_weights={},
        gap_response=_gap_response(gaps, summary),
    )

    response = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))

    assert response.privacy_health_score == 50
    assert response.components["raw_ratio"] == 0.5
    assert response.components["conflict_penalty_applied"] is False
    assert "1 requirement(s) where policy language conflicts" in response.score_assessment
    assert response.score_breakdown["by_jurisdiction"] == {"CA": 100, "NY": 0}


def test_score_clamps_when_partial_credit_is_outside_zero_to_one() -> None:
    partial = [_gap("CA", "general notice", "partial")]
    summary = GapSummary(total_requirements=1, partial=1)
    high, _, _, _ = _service(
        partial_credit=2.0,
        config_weights={},
        gap_response=_gap_response(partial, summary),
    )
    high_response = asyncio.run(high.health_score(HealthScoreRequest(policy_document_id="pol-1")))
    assert high_response.privacy_health_score == 100
    assert high_response.components["raw_ratio"] == 2.0
    assert high_response.score_breakdown["by_jurisdiction"] == {"CA": 100}

    low, _, _, _ = _service(
        partial_credit=-1.0,
        config_weights={},
        gap_response=_gap_response(partial, summary),
    )
    low_response = asyncio.run(low.health_score(HealthScoreRequest(policy_document_id="pol-1")))
    assert low_response.privacy_health_score == 0
    assert low_response.components["raw_ratio"] == -1.0
    assert low_response.score_breakdown["by_jurisdiction"] == {"CA": 0}
    assert low_response.score_assessment.startswith("Critical.")


def test_no_gaps_returns_no_applicable_statutes_without_writing() -> None:
    service, _, v4, storage = _service(
        gap_response=_gap_response([], GapSummary(), company_name="Acme"),
    )

    response = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))

    assert v4.requests
    assert response.privacy_health_score is None
    assert response.error == "no_applicable_statutes"
    assert response.components == {
        "requirements_total": 0,
        "addressed": 0,
        "missing": 0,
        "conflicts": 0,
    }
    assert response.score_breakdown == {}
    assert response.score_assessment.startswith("Unable to assess")
    assert storage.docs == []


def test_failed_analyses_only_are_insufficient_and_copy_summary_counts() -> None:
    gaps = [
        _gap("CA", "one", "addressed", analysis_failed=True),
        _gap("NY", "two", "missing", analysis_failed=True),
    ]
    summary = GapSummary(
        total_requirements=9,
        addressed=5,
        missing=7,
        conflicts=8,
        analysis_failures=2,
    )
    service, _, _, storage = _service(gap_response=_gap_response(gaps, summary))

    response = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))

    assert response.privacy_health_score is None
    assert response.error == "insufficient_analysis"
    assert response.components == {
        "requirements_total": 2,
        "addressed": 5,
        "missing": 7,
        "conflicts": 8,
    }
    assert response.score_assessment.startswith("Unable to assess")
    assert storage.docs == []


def test_save_results_false_skips_persistence() -> None:
    gaps = [_gap("CA", "general notice", "addressed")]
    summary = GapSummary(total_requirements=1, addressed=1)
    service, _, v4, storage = _service(
        config_weights={},
        gap_response=_gap_response(gaps, summary),
    )

    response = asyncio.run(
        service.health_score(HealthScoreRequest(policy_document_id="pol-1", save_results=False))
    )

    assert response.privacy_health_score == 100
    assert v4.requests[0].save_results is False
    assert storage.docs == []


def test_persistence_failure_still_returns_the_score() -> None:
    gaps = [_gap("CA", "general notice", "addressed")]
    summary = GapSummary(total_requirements=1, addressed=1)
    service, _, _, storage = _service(
        config_weights={},
        fail_write=True,
        gap_response=_gap_response(gaps, summary),
    )

    response = asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))

    assert response.privacy_health_score == 100
    assert storage.docs == []


def test_rate_limit_runs_before_gap_analysis() -> None:
    service, limiter, v4, storage = _service(allowed=False)

    try:
        asyncio.run(service.health_score(HealthScoreRequest(policy_document_id="pol-1")))
    except ComplianceSuiteServiceError as exc:
        assert exc.status_code == 429
        assert str(exc) == "Rate limit exceeded"
    else:
        raise AssertionError("expected ComplianceSuiteServiceError")

    assert limiter.calls == 1
    assert v4.requests == []
    assert storage.docs == []
