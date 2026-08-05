"""Regression tests for health-score assessment text banding and notes."""
from __future__ import annotations

import pytest

from compliance_suite_schemas import _score_assessment


def _components(
    *,
    total: int = 10,
    addressed: int = 5,
    missing: int = 3,
    conflicts: int = 0,
    partial: int = 2,
    ambiguous: int = 0,
) -> dict:
    return {
        "requirements_total": total,
        "addressed": addressed,
        "missing": missing,
        "conflicts": conflicts,
        "partial": partial,
        "ambiguous": ambiguous,
    }


@pytest.mark.parametrize(
    ("score", "error"),
    [
        (None, None),
        (None, "no_applicable_statutes"),
        (80, "insufficient_analysis"),
    ],
)
def test_score_assessment_error_or_missing_score(score, error) -> None:
    text = _score_assessment(score, error)
    assert text.startswith("Unable to assess the privacy health score.")
    assert "policy_legal_embeddings" in text


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (100, "Excellent"),
        (90, "Excellent"),
        (89, "Good"),
        (75, "Good"),
        (74, "Fair"),
        (50, "Fair"),
        (49, "Needs improvement"),
        (25, "Needs improvement"),
        (24, "Critical"),
        (0, "Critical"),
    ],
)
def test_score_assessment_band_thresholds(score: int, level: str) -> None:
    text = _score_assessment(score, None, _components())
    assert text.startswith(f"{level}.")


def test_score_assessment_uses_component_counts_when_total_present() -> None:
    text = _score_assessment(80, None, _components(total=12, addressed=9, partial=1))
    assert "addresses 9 of 12 requirements" in text
    assert "1 partially addressed" in text


def test_score_assessment_falls_back_when_total_is_zero() -> None:
    text = _score_assessment(80, None, _components(total=0, addressed=0, partial=0, missing=0))
    assert text.startswith("Good.")
    assert "solid coverage" in text
    assert "of 0 requirements" not in text


def test_score_assessment_includes_conflict_note() -> None:
    text = _score_assessment(60, None, _components(conflicts=2))
    assert "2 requirement(s) where policy language conflicts" in text
    assert "Resolve these conflicts" in text


def test_score_assessment_omits_conflict_note_when_zero() -> None:
    text = _score_assessment(60, None, _components(conflicts=0))
    assert "conflicts with statute expectations" not in text


def test_score_assessment_ambiguous_note_only_below_good() -> None:
    fair = _score_assessment(60, None, _components(ambiguous=3))
    assert "3 requirement(s) were marked ambiguous" in fair

    good = _score_assessment(80, None, _components(ambiguous=3))
    excellent = _score_assessment(95, None, _components(ambiguous=3))
    assert "marked ambiguous" not in good
    assert "marked ambiguous" not in excellent


def test_score_assessment_handles_missing_components() -> None:
    text = _score_assessment(10, None, None)
    assert text.startswith("Critical.")
    assert "critical gaps" in text
