"""Regression tests for ComplianceSuiteService.applicability.

Route tests mock this method. The service decides whether text or a stored
policy is sent to the model, and it copies the model payload into a response
that rejects values the LLM client otherwise keeps.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import pytest
from pydantic import ValidationError

from compliance_config import load_config
from compliance_suite_schemas import ApplicabilityRequest
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _run(coro):
    return asyncio.run(coro)


def _service(find_one=None, llm_payload: Optional[Dict[str, Any]] = None, *, allow: bool = True):
    cfg = load_config()
    cfg.compliance_database = "privacy-compliance"
    cfg.policies_collection = "policies"
    cfg.policy_document_id_field = "document_id"

    policies = MagicMock()
    policies.find_one.side_effect = find_one or (lambda _query: None)
    database = MagicMock()
    database.__getitem__.return_value = policies
    mongo = MagicMock()
    mongo.__getitem__.return_value = database

    llm = MagicMock()
    llm.applicability = AsyncMock(return_value=llm_payload if llm_payload is not None else {})
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=allow)

    service = ComplianceSuiteService(
        mongo_client=mongo,
        config=cfg,
        retriever=MagicMock(),
        llm_client=llm,
        storage=MagicMock(),
        rate_limiter=limiter,
    )
    return service, llm, limiter, mongo, policies


def test_rate_limit_and_blank_text_stop_before_the_model() -> None:
    service, llm, limiter, mongo, _policies = _service(allow=False)

    with pytest.raises(ComplianceSuiteServiceError) as limited:
        _run(service.applicability(ApplicabilityRequest(text="keep me")))

    assert limited.value.status_code == 429
    assert llm.applicability.await_count == 0
    mongo.__getitem__.assert_not_called()
    limiter.allow.assert_awaited()

    blank, blank_llm, _limiter, blank_mongo, _blank_policies = _service()
    with pytest.raises(ComplianceSuiteServiceError) as empty:
        _run(blank.applicability(ApplicabilityRequest(text="   ")))
    assert empty.value.status_code == 400
    assert "empty" in str(empty.value).lower()
    assert blank_llm.applicability.await_count == 0
    blank_mongo.__getitem__.assert_not_called()


def test_text_is_stripped_and_document_lookup_is_skipped() -> None:
    service, llm, _limiter, mongo, _policies = _service(
        llm_payload={"applicable_jurisdictions": ["CA"], "confidence": {"CA": 0.5}}
    )

    response = _run(service.applicability(ApplicabilityRequest(text="  hello  ", policy_document_id="pol-1")))

    assert response.applicable_jurisdictions == ["CA"]
    assert response.confidence == {"CA": 0.5}
    llm.applicability.assert_awaited_once_with("hello")
    mongo.__getitem__.assert_not_called()


def test_empty_text_loads_the_document_and_none_jurisdictions_become_empty() -> None:
    def find_one(query: Dict[str, Any]):
        assert query == {"document_id": "pol-1"}
        return {"text": "  stored policy  ", "company_name": "Acme"}

    service, llm, _limiter, mongo, policies = _service(
        find_one=find_one,
        llm_payload={"applicable_jurisdictions": None, "confidence": None},
    )

    response = _run(
        service.applicability(
            ApplicabilityRequest(
                text="",
                policy_document_id="pol-1",
                database="other-db",
                policy_collection="other_policies",
            )
        )
    )

    assert response.applicable_jurisdictions == []
    assert response.confidence is None
    llm.applicability.assert_awaited_once_with("stored policy")
    mongo.__getitem__.assert_called_with("other-db")
    mongo["other-db"].__getitem__.assert_called_with("other_policies")
    assert policies.find_one.called


def test_missing_policy_is_400_and_null_jurisdiction_items_fail_response_validation() -> None:
    service, llm, _limiter, _mongo, _policies = _service()
    with pytest.raises(ComplianceSuiteServiceError) as missing:
        _run(service.applicability(ApplicabilityRequest(policy_document_id="missing")))
    assert missing.value.status_code == 400
    assert llm.applicability.await_count == 0

    crashing, crashing_llm, _limiter, _mongo2, _policies2 = _service(
        find_one=lambda _query: {"text": "stored"},
        llm_payload={"applicable_jurisdictions": ["CA", None], "confidence": {"CA": 1.0}},
    )
    with pytest.raises(ValidationError):
        _run(crashing.applicability(ApplicabilityRequest(policy_document_id="pol-1")))
    crashing_llm.applicability.assert_awaited()

    bad_confidence, _llm2, _limiter2, _mongo3, _policies3 = _service(
        llm_payload={"applicable_jurisdictions": ["CA"], "confidence": "high"}
    )
    with pytest.raises(ValidationError):
        _run(bad_confidence.applicability(ApplicabilityRequest(text="hello")))
