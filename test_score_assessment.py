"""Tests for the customer-facing privacy health score narrative."""
from __future__ import annotations

from compliance_suite_schemas import _score_assessment


def _components(**overrides: object) -> dict:
    base = {
        "requirements_total": 10,
        "addressed": 8,
        "missing": 1,
        "conflicts": 0,
        "partial": 1,
        "ambiguous": 0,
    }
    base.update(overrides)
    return base


def test_error_or_missing_score_hides_the_band() -> None:
    blocked = _score_assessment(95, "no_applicable_statutes", _components())
    assert blocked.startswith("Unable to assess the privacy health score.")
    assert "Excellent" not in blocked

    missing = _score_assessment(None, None, _components(requirements_total=4, addressed=4))
    assert missing.startswith("Unable to assess the privacy health score.")

    empty_error = _score_assessment(
        10,
        "",
        _components(requirements_total=4, addressed=0, missing=4, partial=0),
    )
    assert empty_error.startswith("Critical.")


def test_score_band_boundaries() -> None:
    components = _components()
    assert _score_assessment(90, None, components).startswith("Excellent.")
    assert _score_assessment(89, None, components).startswith("Good.")
    assert _score_assessment(75, None, components).startswith("Good.")
    assert _score_assessment(74, None, components).startswith("Fair.")
    assert _score_assessment(50, None, components).startswith("Fair.")
    assert _score_assessment(49, None, components).startswith("Needs improvement.")
    assert _score_assessment(25, None, components).startswith("Needs improvement.")
    assert _score_assessment(24, None, components).startswith("Critical.")
    assert _score_assessment(0, None, components).startswith("Critical.")


def test_zero_total_uses_generic_reasoning_and_missing_partial_is_zero() -> None:
    generic = _score_assessment(
        95,
        None,
        {"requirements_total": 0, "addressed": 3, "missing": 0, "conflicts": 0},
    )
    assert "demonstrates strong alignment" in generic
    assert "fully addresses" not in generic

    counted = _score_assessment(
        95,
        None,
        _components(requirements_total=4, addressed=4, missing=0, partial=0),
    )
    assert "fully addresses 4 of 4 applicable privacy requirements." in counted

    good = _score_assessment(
        80,
        None,
        {"requirements_total": 5, "addressed": 4, "missing": 1, "conflicts": 0},
    )
    assert "addresses 4 of 5 requirements, with 0 partially addressed." in good


def test_conflict_note_is_kept_and_ambiguous_note_is_suppressed_above_fair() -> None:
    excellent = _score_assessment(
        95,
        None,
        _components(conflicts=2, ambiguous=3, requirements_total=10, addressed=10, missing=0, partial=0),
    )
    assert "There are 2 requirement(s) where policy language conflicts" in excellent
    assert "marked ambiguous" not in excellent

    good = _score_assessment(
        80,
        None,
        _components(conflicts=0, ambiguous=2, partial=3, requirements_total=10, addressed=6, missing=1),
    )
    assert "with 3 partially addressed." in good
    assert "marked ambiguous" not in good

    fair = _score_assessment(
        60,
        None,
        _components(
            conflicts=1,
            ambiguous=2,
            partial=1,
            missing=3,
            addressed=4,
            requirements_total=10,
        ),
    )
    assert fair.startswith("Fair.")
    assert "with 1 partially addressed and 3 missing." in fair
    assert "There are 1 requirement(s) where policy language conflicts" in fair
    assert "2 requirement(s) were marked ambiguous" in fair


def test_lower_bands_count_missing_requirements_and_none_components_still_score() -> None:
    needs = _score_assessment(
        30,
        None,
        _components(requirements_total=8, addressed=1, missing=5, partial=2, conflicts=0, ambiguous=1),
    )
    assert needs.startswith("Needs improvement.")
    assert "addresses 1 of 8 requirements, with 5 missing and 2 partially addressed." in needs
    assert "1 requirement(s) were marked ambiguous" in needs

    critical = _score_assessment(
        10,
        None,
        _components(requirements_total=8, addressed=0, missing=8, partial=0, ambiguous=0),
    )
    assert "addresses 0 of 8 requirements, with 8 missing." in critical
    assert "partially addressed" not in critical
    assert "marked ambiguous" not in critical

    unscored_components = _score_assessment(80, None, None)
    assert unscored_components.startswith("Good.")
    assert "solid coverage of key privacy obligations." in unscored_components
