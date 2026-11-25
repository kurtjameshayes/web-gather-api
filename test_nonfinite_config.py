"""Regression tests for numeric config values Python accepts but scoring does not clamp."""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import pytest

from compliance_config import load_config
from compliance_evaluator import ComplianceEvaluator
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreRequest,
)
from compliance_suite_service import ComplianceSuiteService
from rate_limiter import RateLimiter
from vector_retriever import StatuteCandidate


_CONFIG_ENV_KEYS = (
    "COMPLIANCE_CONFIG_PATH",
    "COMPLIANCE_TOP_K_STATUTES",
    "COMPLIANCE_EVIDENCE_SCORE_THRESHOLD",
    "COMPLIANCE_MAX_STATUTE_CHARS",
    "COMPLIANCE_MAX_SECTION_CHARS",
    "COMPLIANCE_USE_EMBEDDINGS_COLLECTION",
    "COMPLIANCE_CACHE_SIZE",
    "COMPLIANCE_CACHE_TTL_SECONDS",
    "COMPLIANCE_LLM_MAX_TOKENS",
    "COMPLIANCE_LLM_CONCURRENCY",
    "COMPLIANCE_RATE_LIMIT_PER_MINUTE",
    "COMPLIANCE_ENABLE_REDACTION",
    "COMPLIANCE_ENABLE_AUDIT_LOGGING",
    "COMPLIANCE_ALLOW_RAW_AUDIT",
    "COMPLIANCE_API_KEY",
    "API_KEY",
    "COMPLIANCE_AUTH_REQUIRED",
    "EMBEDDING_MODEL_NAME",
    "LLM_MODEL_NAME",
    "COMPLIANCE_ALLOWED_ROLES",
    "COMPLIANCE_THRESHOLD_COMPLIANT",
    "COMPLIANCE_THRESHOLD_NON_COMPLIANT",
    "COMPLIANCE_MAX_STATUTE_QUOTE_CHARS",
    "COMPLIANCE_CONFLICT_PENALTY",
    "COMPLIANCE_PARTIAL_CREDIT_PERCENT",
    "GAP_ANALYSIS_V3_TOP_K",
    "GAP_ANALYSIS_V3_NUM_CANDIDATES",
    "GAP_ANALYSIS_V3_SCORE_THRESHOLD",
    "ADAPTIVE_FEEDBACK_ENABLED",
    "ADAPTIVE_FEEDBACK_MAX_ITEMS",
)


def _load_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, updates: dict[str, str]):
    for key in _CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "empty-config.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(path))
    for key, value in updates.items():
        monkeypatch.setenv(key, value)
    return load_config()


def _candidate(score: float) -> StatuteCandidate:
    return StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="CA",
        title="Record keeping",
        section_id="R-1",
        chunk_text="Records are kept for two years.",
        score=score,
        chunk_id="chunk-1",
    )


def _evaluate(evaluator: ComplianceEvaluator, thresholds, compliance: str, confidence: float, evidence: float):
    payload = (
        '{"compliance":"%s","confidence":%s,"rationale":"Reviewed.",'
        '"applied_statutes":[{"statute_id":"stat-1","evidence_score":%s}],'
        '"remediation_suggestions":["Keep the current wording."]}'
        % (compliance, confidence, evidence)
    )
    return evaluator.evaluate(
        section_id="section-1",
        section_text="We keep records for two years.",
        candidates=[_candidate(evidence)],
        llm_response_text=payload,
        thresholds=thresholds,
        explainability=False,
    )


def _gap(status: str) -> GapItem:
    return GapItem(
        jurisdiction="CA",
        statute_reference="§ 1",
        requirement_summary="unrelated obligation",
        status=status,
    )


class _V4:
    def __init__(self, response: GapAnalysisResponse) -> None:
        self.response = response

    async def run(self, req: Any) -> GapAnalysisResponse:
        return self.response


class _Allow:
    async def allow(self) -> bool:
        return True


def _health(config, response: GapAnalysisResponse):
    service = ComplianceSuiteService(
        mongo_client=object(),
        config=config,
        retriever=object(),
        llm_client=object(),
        storage=object(),
        rate_limiter=_Allow(),
        gap_analysis_v4_service=_V4(response),
    )
    return asyncio.run(
        service.health_score(
            HealthScoreRequest(policy_document_id="pol-1", save_results=False)
        )
    )


def test_nonfinite_and_underscore_values_are_stored(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {
            "COMPLIANCE_THRESHOLD_COMPLIANT": "nan",
            "COMPLIANCE_THRESHOLD_NON_COMPLIANT": "-inf",
            "COMPLIANCE_EVIDENCE_SCORE_THRESHOLD": "1e2",
            "COMPLIANCE_PARTIAL_CREDIT_PERCENT": "Infinity",
            "COMPLIANCE_CONFLICT_PENALTY": "NaN",
            "COMPLIANCE_TOP_K_STATUTES": "1_000",
            "GAP_ANALYSIS_V3_TOP_K": "+8",
            "COMPLIANCE_LLM_MAX_TOKENS": "08",
            "COMPLIANCE_RATE_LIMIT_PER_MINUTE": "-5",
            "COMPLIANCE_CACHE_SIZE": "1e2",
            "GAP_ANALYSIS_V3_SCORE_THRESHOLD": "1_000.5",
            "COMPLIANCE_ENABLE_REDACTION": "y",
            "COMPLIANCE_ENABLE_AUDIT_LOGGING": "on",
            "COMPLIANCE_ALLOW_RAW_AUDIT": "Y",
        },
    )

    assert math.isnan(config.confidence_thresholds.compliant)
    assert config.confidence_thresholds.non_compliant == float("-inf")
    assert config.evidence_score_threshold == 100.0
    assert math.isinf(config.partial_credit_percent) and config.partial_credit_percent > 0
    assert math.isnan(config.conflict_penalty_multiplier)
    assert config.top_k_statutes == 1000
    assert config.gap_analysis_v3_top_k == 8
    assert config.llm_max_tokens == 8
    assert config.rate_limit_per_minute == -5
    assert config.cache_size == 512
    assert config.gap_analysis_v3_score_threshold == 1000.5
    assert config.enable_redaction is True
    assert config.enable_audit_logging is True
    assert config.allow_raw_audit is True


def test_nan_and_negative_infinity_thresholds_change_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {
            "COMPLIANCE_THRESHOLD_COMPLIANT": "nan",
            "COMPLIANCE_THRESHOLD_NON_COMPLIANT": "-inf",
        },
    )
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)

    confident = _evaluate(
        evaluator,
        config.confidence_thresholds,
        compliance="compliant",
        confidence=1.0,
        evidence=0.9,
    )
    assert confident.compliance == "neither"
    assert confident.warnings == []

    unconfident = _evaluate(
        evaluator,
        config.confidence_thresholds,
        compliance="non_compliant",
        confidence=0.0,
        evidence=0.9,
    )
    assert unconfident.compliance == "non_compliant"
    assert unconfident.warnings == []


def test_infinite_evidence_threshold_downgrades_a_strong_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_EVIDENCE_SCORE_THRESHOLD": "inf"},
    )
    evaluator = ComplianceEvaluator(evidence_score_threshold=config.evidence_score_threshold)
    result = _evaluate(
        evaluator,
        config.confidence_thresholds,
        compliance="compliant",
        confidence=0.99,
        evidence=0.99,
    )
    assert result.compliance == "neither"
    assert result.warnings == ["Low retrieval evidence score; marked as neither."]


def test_nan_evidence_threshold_keeps_a_zero_evidence_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_EVIDENCE_SCORE_THRESHOLD": "nan"},
    )
    evaluator = ComplianceEvaluator(evidence_score_threshold=config.evidence_score_threshold)
    result = _evaluate(
        evaluator,
        config.confidence_thresholds,
        compliance="compliant",
        confidence=0.99,
        evidence=0.0,
    )
    assert result.compliance == "compliant"
    assert result.warnings == []


def test_nan_conflict_penalty_does_not_change_the_health_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_CONFLICT_PENALTY": "NaN"},
    )
    response = GapAnalysisResponse(
        policy_document_id="pol-1",
        analyzed_at="2026-01-01T00:00:00+00:00",
        gaps=[_gap("addressed"), _gap("conflict")],
        summary=GapSummary(total_requirements=2, addressed=1, conflicts=1),
    )
    scored = _health(config, response)
    assert scored.privacy_health_score == 50
    assert scored.components["raw_ratio"] == 0.5
    assert scored.components["conflict_penalty_applied"] is False


def test_nonfinite_partial_credit_aborts_the_health_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    response = GapAnalysisResponse(
        policy_document_id="pol-1",
        analyzed_at="2026-01-01T00:00:00+00:00",
        gaps=[_gap("partial")],
        summary=GapSummary(total_requirements=1, partial=1),
    )

    nan_config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_PARTIAL_CREDIT_PERCENT": "nan"},
    )
    with pytest.raises(ValueError):
        _health(nan_config, response)

    inf_config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_PARTIAL_CREDIT_PERCENT": "inf"},
    )
    with pytest.raises(OverflowError):
        _health(inf_config, response)


def test_negative_infinite_conflict_penalty_aborts_the_health_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_CONFLICT_PENALTY": "-inf"},
    )
    response = GapAnalysisResponse(
        policy_document_id="pol-1",
        analyzed_at="2026-01-01T00:00:00+00:00",
        gaps=[_gap("addressed"), _gap("conflict")],
        summary=GapSummary(total_requirements=2, addressed=1, conflicts=1),
    )
    with pytest.raises(OverflowError):
        _health(config, response)


def test_negative_rate_limit_still_allows_one_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    config = _load_isolated(
        monkeypatch,
        tmp_path,
        {"COMPLIANCE_RATE_LIMIT_PER_MINUTE": "-5"},
    )
    now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    limiter = RateLimiter(config.rate_limit_per_minute)
    assert limiter._rate == 1
    assert asyncio.run(limiter.allow()) is True
    assert asyncio.run(limiter.allow()) is False

    zero = RateLimiter(0)
    assert zero._rate == 1
