"""Regression tests for ComplianceEvaluator edge cases and rule checks."""
from __future__ import annotations

import json

from compliance_config import ConfidenceThresholds
from compliance_evaluator import ComplianceEvaluator
from vector_retriever import StatuteCandidate


THRESHOLDS = ConfidenceThresholds(compliant=0.75, non_compliant=0.75)


def _candidate(
    *,
    statute_id: str = "stat-1",
    chunk_text: str = "Data retention must not exceed 3 years.",
    score: float = 0.9,
) -> StatuteCandidate:
    return StatuteCandidate(
        statute_id=statute_id,
        jurisdiction="US",
        title="Privacy Requirement",
        section_id="R-1",
        chunk_text=chunk_text,
        score=score,
        chunk_id="chunk-1",
    )


def _payload(**overrides: object) -> str:
    base = {
        "section_id": "section_1",
        "applied_statutes": [
            {
                "statute_id": "stat-1",
                "jurisdiction": "US",
                "title": "Privacy Requirement",
                "matched_span": "retain",
                "evidence_score": 0.9,
            }
        ],
        "compliance": "compliant",
        "confidence": 0.9,
        "rationale": "Policy language matches the statute.",
        "remediation_suggestions": ["Keep retention under 3 years."],
    }
    base.update(overrides)
    return json.dumps(base)


def test_consent_rule_downgrades_compliant_when_policy_omits_consent() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We share data with partners for analytics.",
        candidates=[_candidate(chunk_text="Explicit consent or opt-in is required for sharing.")],
        llm_response_text=_payload(),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "non_compliant"
    assert any("Consent requirement" in w for w in result.warnings)


def test_consent_rule_keeps_compliant_when_policy_mentions_consent() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We obtain opt-in consent before sharing data.",
        candidates=[_candidate(chunk_text="Explicit consent or opt-in is required for sharing.")],
        llm_response_text=_payload(),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "compliant"


def test_invalid_compliance_value_defaults_to_neither() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate()],
        llm_response_text=_payload(compliance="maybe"),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "neither"
    assert any("invalid compliance" in w.lower() for w in result.warnings)


def test_empty_llm_response_defaults_safely() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate()],
        llm_response_text="",
        thresholds=THRESHOLDS,
        explainability=True,
    )
    assert result.compliance == "neither"
    assert result.rationale
    assert result.remediation_suggestions
    assert any("empty" in w.lower() for w in result.warnings)
    assert result.retrieval_trace


def test_embedded_json_block_is_parsed() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    wrapped = "Here is the analysis:\n" + _payload() + "\nThanks."
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate()],
        llm_response_text=wrapped,
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "compliant"
    assert result.applied_statutes[0]["statute_id"] == "stat-1"


def test_malformed_json_yields_neither_with_warning() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate()],
        llm_response_text="not json at all",
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "neither"
    assert result.warnings


def test_low_evidence_score_forces_neither() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.8)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate(score=0.5)],
        llm_response_text=_payload(
            applied_statutes=[
                {
                    "statute_id": "stat-1",
                    "jurisdiction": "US",
                    "title": "Privacy Requirement",
                    "matched_span": "retain",
                    "evidence_score": 0.4,
                }
            ]
        ),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "neither"
    assert any("Low retrieval evidence" in w for w in result.warnings)


def test_empty_applied_statutes_falls_back_to_candidates() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data for analytics.",
        candidates=[_candidate(statute_id="fallback-stat", score=0.85)],
        llm_response_text=_payload(
            applied_statutes=[],
            compliance="neither",
            confidence=0.4,
        ),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert len(result.applied_statutes) == 1
    assert result.applied_statutes[0]["statute_id"] == "fallback-stat"
    assert result.applied_statutes[0]["evidence_score"] == 0.85


def test_non_compliant_requires_confidence_threshold() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    result = evaluator.evaluate(
        section_id="section_1",
        section_text="We retain user data forever.",
        candidates=[_candidate()],
        llm_response_text=_payload(compliance="non_compliant", confidence=0.5),
        thresholds=THRESHOLDS,
        explainability=False,
    )
    assert result.compliance == "neither"
