"""Rule checks that can override an LLM compliance label."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import ConfidenceThresholds
from compliance_evaluator import ComplianceEvaluator
from vector_retriever import StatuteCandidate


THRESHOLDS = ConfidenceThresholds(compliant=0.75, non_compliant=0.75)


def _candidate(chunk_text: str, score: float = 0.9) -> StatuteCandidate:
    return StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="CA",
        title="Consent",
        section_id="S-1",
        chunk_text=chunk_text,
        score=score,
        chunk_id="c1",
    )


def _evaluate(
    evaluator: ComplianceEvaluator,
    payload: dict,
    *,
    section_text: str,
    candidates: list[StatuteCandidate],
    explainability: bool = True,
    raw_text: str | None = None,
):
    return evaluator.evaluate(
        section_id="section-1",
        section_text=section_text,
        candidates=candidates,
        llm_response_text=raw_text if raw_text is not None else json.dumps(payload),
        thresholds=THRESHOLDS,
        explainability=explainability,
    )


def test_consent_rule_demotes_only_a_confident_compliant_label() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidate = _candidate("Explicit consent is required before sharing.")
    payload = {
        "applied_statutes": [{"statute_id": "stat-1", "evidence_score": 0.9}],
        "compliance": "compliant",
        "confidence": 0.95,
        "rationale": "The policy matches.",
        "remediation_suggestions": ["None needed"],
    }

    demoted = _evaluate(
        evaluator,
        payload,
        section_text="We share data with partners.",
        candidates=[candidate],
    )
    assert demoted.compliance == "non_compliant"
    assert "Consent requirement detected without corresponding policy language." in demoted.warnings

    kept = _evaluate(
        evaluator,
        payload,
        section_text="We share data only with Opt-In consent.",
        candidates=[candidate],
    )
    assert kept.compliance == "compliant"
    assert kept.warnings == []

    opt_in_statute = _evaluate(
        evaluator,
        payload,
        section_text="We share data with partners.",
        candidates=[_candidate("Sale requires an opt-in.")],
    )
    assert opt_in_statute.compliance == "non_compliant"

    no_consent_statute = _evaluate(
        evaluator,
        payload,
        section_text="We share data with partners.",
        candidates=[_candidate("Retention must not exceed three years.")],
    )
    assert no_consent_statute.compliance == "compliant"

    # Below the confidence gate the label is already neither, so the consent warning is not added.
    payload["confidence"] = 0.5
    low_confidence = _evaluate(
        evaluator,
        payload,
        section_text="We share data with partners.",
        candidates=[candidate],
    )
    assert low_confidence.compliance == "neither"
    assert low_confidence.warnings == []

    # "opt in" without the hyphen does not satisfy the policy-language check.
    payload["confidence"] = 0.95
    hyphen = _evaluate(
        evaluator,
        payload,
        section_text="Users may opt in later.",
        candidates=[_candidate("An opt-in is required.")],
    )
    assert hyphen.compliance == "non_compliant"


def test_low_evidence_overrides_both_compliant_and_non_compliant() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    payload = {
        "applied_statutes": [{"statute_id": "stat-1", "evidence_score": 0.05}],
        "compliance": "non_compliant",
        "confidence": 0.99,
        "rationale": "Clear conflict.",
        "remediation_suggestions": ["Remove the clause."],
    }
    result = _evaluate(
        evaluator,
        payload,
        section_text="We retain data forever.",
        candidates=[_candidate("Retention must not exceed three years.", score=0.95)],
    )
    assert result.compliance == "neither"
    assert "Low retrieval evidence score; marked as neither." in result.warnings
    assert result.applied_statutes[0]["evidence_score"] == 0.05

    # A 0.0 evidence score is falsy and falls back to the candidate score.
    payload["applied_statutes"] = [{"statute_id": "stat-1", "evidence_score": 0.0}]
    fallback = _evaluate(
        evaluator,
        payload,
        section_text="We retain data forever.",
        candidates=[_candidate("Retention must not exceed three years.", score=0.4)],
    )
    assert fallback.applied_statutes[0]["evidence_score"] == 0.4
    assert fallback.compliance == "non_compliant"
    assert "Low retrieval evidence score" not in " ".join(fallback.warnings)


def test_invalid_or_empty_llm_output_defaults_and_candidate_fallback() -> None:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    candidate = _candidate("Retention must not exceed three years.", score=0.8)

    invalid = _evaluate(
        evaluator,
        {
            "compliance": "yes",
            "confidence": 0.99,
            "rationale": "  ",
            "remediation_suggestions": ["  ", "Add a retention limit."],
            "applied_statutes": [],
        },
        section_text="We keep data.",
        candidates=[candidate],
        explainability=False,
    )
    assert invalid.compliance == "neither"
    assert "LLM returned invalid compliance value; defaulted to neither." in invalid.warnings
    assert invalid.rationale == "Insufficient evidence to determine compliance with confidence."
    assert invalid.remediation_suggestions == ["Add a retention limit."]
    assert invalid.retrieval_trace == []
    assert invalid.applied_statutes[0]["statute_id"] == "stat-1"
    assert invalid.applied_statutes[0]["evidence_score"] == 0.8

    empty = _evaluate(
        evaluator,
        {},
        section_text="We keep data.",
        candidates=[candidate],
        raw_text="",
    )
    assert empty.compliance == "neither"
    assert "LLM response was empty." in empty.warnings
    assert empty.remediation_suggestions == [
        "Clarify policy language and align with statutory requirements."
    ]
    assert empty.retrieval_trace == ["stat-1:c1:0.8000"]

    prose = _evaluate(
        evaluator,
        {},
        section_text="We keep data.",
        candidates=[candidate],
        raw_text='Note: {"compliance": "non_compliant", "confidence": 0.9, "rationale": "missing limit"}',
    )
    assert prose.compliance == "non_compliant"
    assert prose.rationale == "missing limit"
    assert prose.warnings == []

    broken = _evaluate(
        evaluator,
        {},
        section_text="We keep data.",
        candidates=[candidate],
        raw_text="prefix {not json} suffix",
    )
    assert "Failed to parse LLM JSON response." in broken.warnings

    no_object = _evaluate(
        evaluator,
        {},
        section_text="We keep data.",
        candidates=[candidate],
        raw_text="no json here",
    )
    assert "No JSON object found in LLM response." in no_object.warnings

    high = _evaluate(
        evaluator,
        {"compliance": "compliant", "confidence": 1.5, "rationale": "ok", "applied_statutes": []},
        section_text="We keep data.",
        candidates=[candidate],
    )
    assert high.confidence == 1.0
    assert high.compliance == "compliant"
