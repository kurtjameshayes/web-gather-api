"""Regression tests for compliance suite request-model validation guards."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from compliance_suite_schemas import (
    ApplicabilityRequest,
    ConsumerRightsRouterRequest,
    GapAnalysisRequest,
    MultiJurisdictionalRequest,
    ReportRequest,
    VALID_REQUEST_TYPES,
)


def test_applicability_request_requires_policy_source() -> None:
    with pytest.raises(ValidationError, match="policy_document_id or text"):
        ApplicabilityRequest()


def test_applicability_request_accepts_text_or_document_id() -> None:
    by_text = ApplicabilityRequest(text="We collect personal data.")
    by_id = ApplicabilityRequest(policy_document_id="policy-1")
    assert by_text.text.startswith("We collect")
    assert by_id.policy_document_id == "policy-1"


def test_gap_analysis_request_requires_policy_source() -> None:
    with pytest.raises(ValidationError, match="policy_document_id or policy_document_ids"):
        GapAnalysisRequest()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_document_id": "   "},
        {"policy_document_ids": []},
        {"policy_document_id": "", "policy_document_ids": []},
    ],
)
def test_gap_analysis_request_rejects_empty_sources(kwargs: dict) -> None:
    with pytest.raises(ValidationError, match="policy_document_id or policy_document_ids"):
        GapAnalysisRequest(**kwargs)


def test_gap_analysis_request_accepts_single_or_list_ids() -> None:
    single = GapAnalysisRequest(policy_document_id="policy-1")
    multi = GapAnalysisRequest(policy_document_ids=["a", "b"])
    assert single.policy_document_id == "policy-1"
    assert multi.policy_document_ids == ["a", "b"]
    assert single.run_async is True
    assert single.save_results is True


def test_consumer_rights_router_request_requires_policy_source() -> None:
    with pytest.raises(ValidationError, match="policy_document_id or text"):
        ConsumerRightsRouterRequest()


def test_consumer_rights_router_request_rejects_invalid_types() -> None:
    with pytest.raises(ValidationError, match="Invalid request_types"):
        ConsumerRightsRouterRequest(text="policy", request_types=["deletion", "unknown"])


def test_consumer_rights_router_request_accepts_valid_types() -> None:
    req = ConsumerRightsRouterRequest(
        text="policy body",
        request_types=list(VALID_REQUEST_TYPES),
    )
    assert req.request_types == list(VALID_REQUEST_TYPES)


def test_multi_jurisdictional_request_requires_at_least_one_jurisdiction() -> None:
    with pytest.raises(ValidationError):
        MultiJurisdictionalRequest(applicable_jurisdictions=[])
    ok = MultiJurisdictionalRequest(applicable_jurisdictions=["CA"])
    assert ok.applicable_jurisdictions == ["CA"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"policy_document_id": "p1", "format": "html"}, "format"),
        (
            {
                "policy_document_id": "p1",
                "format": "markdown",
                "source": "cached",
            },
            "source",
        ),
        ({"policy_document_id": "", "format": "markdown"}, "policy_document_id"),
    ],
)
def test_report_request_rejects_invalid_fields(kwargs: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ReportRequest(**kwargs)


def test_report_request_defaults_and_valid_values() -> None:
    req = ReportRequest(policy_document_id="policy-1", format="pdf")
    assert req.source == "latest_stored"
    assert req.include_gap is True
    assert req.include_health_score is True
    assert req.include_multi_jurisdictional is False
