"""Regression tests for statute-policy compliance request/response schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    ConfidenceThresholds,
    RequestOptions,
    StatutePolicyComplianceRequest,
)


def test_statute_policy_compliance_request_requires_non_empty_ids() -> None:
    req = StatutePolicyComplianceRequest(
        policy_id="policy-1",
        jurisdiction="CA",
        policy_collection="policies",
    )
    assert req.policy_id == "policy-1"
    assert req.jurisdiction == "CA"
    assert req.policy_collection == "policies"

    with pytest.raises(ValidationError):
        StatutePolicyComplianceRequest(policy_id="", jurisdiction="CA")
    with pytest.raises(ValidationError):
        StatutePolicyComplianceRequest(policy_id="policy-1", jurisdiction="")
    with pytest.raises(ValidationError):
        StatutePolicyComplianceRequest(
            policy_id="policy-1",
            jurisdiction="CA",
            policy_collection="",
        )


def test_request_options_and_confidence_threshold_bounds() -> None:
    opts = RequestOptions()
    assert opts.explainability is True
    assert opts.redact_pii is True

    thresholds = ConfidenceThresholds()
    assert thresholds.compliant == 0.75
    assert thresholds.non_compliant == 0.75

    with pytest.raises(ValidationError):
        ConfidenceThresholds(compliant=1.5)
    with pytest.raises(ValidationError):
        ConfidenceThresholds(non_compliant=-0.1)
