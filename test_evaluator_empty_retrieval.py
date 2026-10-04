"""Low-evidence demotion must not run when retrieval returned no candidates."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import ConfidenceThresholds
from compliance_evaluator import ComplianceEvaluator


def _evaluate(payload: dict) -> object:
    evaluator = ComplianceEvaluator(evidence_score_threshold=0.2)
    return evaluator.evaluate(
        section_id="sec-1",
        section_text="We collect personal information.",
        candidates=[],
        llm_response_text=json.dumps(payload),
        thresholds=ConfidenceThresholds(compliant=0.75, non_compliant=0.75),
        explainability=True,
    )


def test_empty_retrieval_keeps_confident_labels_despite_low_evidence() -> None:
    """The override is `if candidates and max_evidence < threshold`.

    With no retrieved candidates, a confident compliant or non_compliant label
    is kept. An explicit 0.01 evidence score does not add the low-evidence
    warning, and the retrieval trace stays empty.
    """
    for label in ("compliant", "non_compliant"):
        result = _evaluate(
            {
                "compliance": label,
                "confidence": 0.95,
                "rationale": "Model label.",
                "applied_statutes": [
                    {
                        "statute_id": "stat-x",
                        "evidence_score": 0.01,
                        "jurisdiction": "CA",
                        "title": "Right to know",
                        "section_id": "1798.100",
                    }
                ],
                "remediation_suggestions": ["Keep the disclosure."],
            }
        )
        assert result.compliance == label
        assert result.warnings == []
        assert result.retrieval_trace == []
        assert result.applied_statutes[0]["evidence_score"] == 0.01
        assert result.applied_statutes[0]["statute_id"] == "stat-x"


def test_empty_retrieval_and_empty_applied_statutes_stay_compliant() -> None:
    """No candidates and no applied statutes still leave max evidence at 0.0.

    That score is under the threshold, but the demotion does not run, so a
    confident compliant label is returned with an empty statute list.
    """
    result = _evaluate(
        {
            "compliance": "compliant",
            "confidence": 0.95,
            "rationale": "No statute was retrieved.",
            "applied_statutes": [],
            "remediation_suggestions": ["Keep the disclosure."],
        }
    )
    assert result.compliance == "compliant"
    assert result.confidence == 0.95
    assert result.applied_statutes == []
    assert result.retrieval_trace == []
    assert result.warnings == []
