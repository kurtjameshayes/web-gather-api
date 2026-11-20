"""Tests for compliance-label rules that can override the model."""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import ConfidenceThresholds
from compliance_evaluator import ComplianceEvaluator
from vector_retriever import StatuteCandidate


def _candidate(chunk_text: str, score: float = 0.9) -> StatuteCandidate:
    return StatuteCandidate(
        statute_id="stat-1",
        jurisdiction="CA",
        title="Consent",
        section_id="C-1",
        chunk_text=chunk_text,
        score=score,
        chunk_id="chunk-1",
    )


def _evaluate(
    payload: dict[str, Any] | str,
    section_text: str,
    candidates: list[StatuteCandidate],
    evidence_score_threshold: float = 0.2,
) -> Any:
    evaluator = ComplianceEvaluator(evidence_score_threshold=evidence_score_threshold)
    response = payload if isinstance(payload, str) else json.dumps(payload)
    return evaluator.evaluate(
        section_id="section_1",
        section_text=section_text,
        candidates=candidates,
        llm_response_text=response,
        thresholds=ConfidenceThresholds(compliant=0.75, non_compliant=0.75),
        explainability=True,
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "compliance": "compliant",
        "confidence": 0.95,
        "rationale": "The policy matches the statute.",
        "remediation_suggestions": ["Keep the current language."],
        "applied_statutes": [
            {
                "statute_id": "stat-1",
                "evidence_score": 0.9,
                "matched_span": "consent",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_confidence_threshold_runs_before_consent_downgrade() -> None:
    result = _evaluate(
        _payload(confidence=0.5),
        "We share personal information with partners.",
        [_candidate("Explicit consent is required before sharing.")],
    )
    assert result.compliance == "neither"
    assert "Consent requirement detected" not in " ".join(result.warnings)


def test_consent_rule_downgrades_only_a_confident_compliant_label() -> None:
    silent_policy = "We share personal information with partners."
    result = _evaluate(
        _payload(),
        silent_policy,
        [_candidate("Explicit CONSENT is required before sharing.")],
    )
    assert result.compliance == "non_compliant"
    assert result.warnings == [
        "Consent requirement detected without corresponding policy language."
    ]

    opted_in = _evaluate(
        _payload(),
        "Customers must Opt-In before we sell data.",
        [_candidate("Sale requires an opt-in.")],
    )
    assert opted_in.compliance == "compliant"
    assert opted_in.warnings == []

    already_non_compliant = _evaluate(
        _payload(compliance="non_compliant"),
        silent_policy,
        [_candidate("Explicit consent is required.")],
    )
    assert already_non_compliant.compliance == "non_compliant"
    assert already_non_compliant.warnings == []


def test_low_evidence_overrides_consent_downgrade_using_model_score() -> None:
    downgraded_then_neither = _evaluate(
        _payload(applied_statutes=[{"statute_id": "stat-1", "evidence_score": 0.05}]),
        "We share personal information with partners.",
        [_candidate("Explicit consent is required.", score=0.05)],
    )
    assert downgraded_then_neither.compliance == "neither"
    assert downgraded_then_neither.warnings == [
        "Consent requirement detected without corresponding policy language.",
        "Low retrieval evidence score; marked as neither.",
    ]

    model_score_wins = _evaluate(
        _payload(applied_statutes=[{"statute_id": "stat-1", "evidence_score": 0.9}]),
        "We retain records for one year.",
        [_candidate("Retention must not exceed three years.", score=0.01)],
    )
    assert model_score_wins.compliance == "compliant"
    assert model_score_wins.warnings == []
    assert model_score_wins.applied_statutes[0]["evidence_score"] == 0.9


def test_invalid_or_wrapped_model_output_and_no_candidates() -> None:
    invalid = _evaluate(
        _payload(compliance="  MOSTLY_COMPLIANT  "),
        "We retain records for one year.",
        [_candidate("Retention must not exceed three years.")],
    )
    assert invalid.compliance == "neither"
    assert "LLM returned invalid compliance value; defaulted to neither." in invalid.warnings

    padded = _evaluate(
        _payload(compliance="  Compliant  "),
        "We retain records for one year.",
        [_candidate("Retention must not exceed three years.")],
    )
    assert padded.compliance == "compliant"

    wrapped = (
        'Analysis follows {"compliance": "compliant", "confidence": 0.9, '
        '"rationale": "Aligned.", "remediation_suggestions": ["Keep it."]} thanks'
    )
    parsed = _evaluate(
        wrapped,
        "We retain records for one year.",
        [_candidate("Retention must not exceed three years.")],
    )
    assert parsed.compliance == "compliant"
    assert parsed.rationale == "Aligned."

    no_candidates = _evaluate(_payload(applied_statutes=[]), "We retain records.", [])
    assert no_candidates.compliance == "compliant"
    assert no_candidates.applied_statutes == []
    assert "Low retrieval evidence score" not in " ".join(no_candidates.warnings)
