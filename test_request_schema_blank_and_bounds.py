"""Leftover request-schema guards for blank ids, row limits, and enums.

Gap analysis already rejects a whitespace-only policy_document_id and an empty
policy_document_ids list. These cases pin the neighboring contracts: blank
entries inside a non-empty list, num_rows coercion, case-sensitive gap enums,
and source checks that do not strip.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from compliance_suite_schemas import (
    ApplicabilityRequest,
    ConsumerRightsRouterRequest,
    GapAnalysisRequest,
    GapItem,
    HealthScoreRequest,
    SuggestPolicyRequest,
)


def test_gap_analysis_accepts_blank_ids_inside_a_non_empty_list() -> None:
    """Only list length is checked, so blank entries still satisfy the source rule.

    A later strip-each-id change would turn these into validation errors and
    reject clients that currently pass the gate and fail later as not-found.
    """
    blank = GapAnalysisRequest(policy_document_ids=[""])
    padded = GapAnalysisRequest(policy_document_ids=["   "])
    mixed = GapAnalysisRequest(policy_document_ids=["", "pol-1"])
    kept = GapAnalysisRequest(policy_document_id="  pol-1  ")

    assert blank.policy_document_ids == [""]
    assert padded.policy_document_ids == ["   "]
    assert mixed.policy_document_ids == ["", "pol-1"]
    assert kept.policy_document_id == "  pol-1  "
    assert kept.policy_document_ids is None


def test_gap_analysis_rejects_non_string_ids_in_the_list() -> None:
    with pytest.raises(ValidationError, match="valid string"):
        GapAnalysisRequest(policy_document_ids=[None])


@pytest.mark.parametrize("num_rows", [0, -1, False])
def test_gap_analysis_num_rows_rejects_non_positive(num_rows: object) -> None:
    """0 and False are not a partial run. False coerces to 0, then fails gt=0.

    The service treats a falsy num_rows as "no cap", so this bound is what
    stops a zero from silently analyzing the full statute set.
    """
    with pytest.raises(ValidationError, match="greater than 0"):
        GapAnalysisRequest(policy_document_id="pol-1", num_rows=num_rows)


def test_gap_analysis_num_rows_coerces_integral_values() -> None:
    """JSON true and whole-number strings become a positive row cap."""
    assert GapAnalysisRequest(policy_document_id="pol-1", num_rows=True).num_rows == 1
    assert GapAnalysisRequest(policy_document_id="pol-1", num_rows="3").num_rows == 3
    assert GapAnalysisRequest(policy_document_id="pol-1", num_rows=2.0).num_rows == 2
    assert GapAnalysisRequest(policy_document_id="pol-1").num_rows is None


def test_gap_analysis_num_rows_rejects_fractional_floats() -> None:
    with pytest.raises(ValidationError, match="fractional part"):
        GapAnalysisRequest(policy_document_id="pol-1", num_rows=1.5)


@pytest.mark.parametrize("status", ["missing", "addressed", "conflict", "partial", "ambiguous"])
def test_gap_item_accepts_exact_status_values(status: str) -> None:
    item = GapItem(
        jurisdiction="CA",
        statute_reference="1798.100",
        requirement_summary="Right to know",
        status=status,
    )
    assert item.status == status
    assert item.confidence is None
    assert item.analysis_failed is False


@pytest.mark.parametrize("status", ["Addressed", " addressed ", "unknown", ""])
def test_gap_item_status_pattern_is_case_sensitive(status: str) -> None:
    with pytest.raises(ValidationError, match="pattern"):
        GapItem(
            jurisdiction="CA",
            statute_reference="1798.100",
            requirement_summary="Right to know",
            status=status,
        )


def test_gap_item_confidence_pattern_is_case_sensitive() -> None:
    item = GapItem(
        jurisdiction="CA",
        statute_reference="1798.100",
        requirement_summary="Right to know",
        status="partial",
        confidence="medium",
    )
    assert item.confidence == "medium"
    with pytest.raises(ValidationError, match="pattern"):
        GapItem(
            jurisdiction="CA",
            statute_reference="1798.100",
            requirement_summary="Right to know",
            status="partial",
            confidence="HIGH",
        )


def test_applicability_and_consumer_rights_keep_whitespace_sources() -> None:
    """These requests do not strip, unlike gap analysis's single-id check."""
    by_id = ApplicabilityRequest(policy_document_id="   ")
    by_text = ApplicabilityRequest(text="   ")
    rights = ConsumerRightsRouterRequest(policy_document_id="   ")

    assert by_id.policy_document_id == "   "
    assert by_id.text is None
    assert by_text.text == "   "
    assert rights.policy_document_id == "   "

    with pytest.raises(ValidationError, match="policy_document_id or text"):
        ApplicabilityRequest(policy_document_id="", text="")
    with pytest.raises(ValidationError, match="policy_document_id or text"):
        ConsumerRightsRouterRequest(policy_document_id="", text="")


def test_consumer_rights_request_types_skip_empty_and_reject_case_variants() -> None:
    """An empty list skips the allow-list. Matching is exact, not case-folded."""
    empty = ConsumerRightsRouterRequest(text="policy body", request_types=[])
    omitted = ConsumerRightsRouterRequest(text="policy body")
    assert empty.request_types == []
    assert omitted.request_types is None

    allowed = ConsumerRightsRouterRequest(text="policy body", request_types=["optout"])
    assert allowed.request_types == ["optout"]

    with pytest.raises(ValidationError, match="Invalid request_types"):
        ConsumerRightsRouterRequest(text="policy body", request_types=["Deletion"])
    with pytest.raises(ValidationError, match="Invalid request_types"):
        ConsumerRightsRouterRequest(text="policy body", request_types=[""])


def test_min_length_fields_treat_a_single_space_as_present() -> None:
    """Empty strings fail min_length. A space is length 1 and is stored as-is."""
    health = HealthScoreRequest(policy_document_id=" ")
    suggest = SuggestPolicyRequest(
        policy_text=" ",
        gap_analysis_text=" ",
        gap_analysis_match=" ",
        statute_text=" ",
    )
    assert health.policy_document_id == " "
    assert suggest.policy_text == " "
    assert suggest.statute_text == " "

    with pytest.raises(ValidationError, match="at least 1"):
        HealthScoreRequest(policy_document_id="")
    with pytest.raises(ValidationError, match="at least 1"):
        SuggestPolicyRequest(
            policy_text="",
            gap_analysis_text="gap",
            gap_analysis_match="match",
            statute_text="statute",
        )
