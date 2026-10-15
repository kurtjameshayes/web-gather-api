"""Regression tests for POST /statute-policy-compliance error mapping.

StatutePolicyComplianceRequest accepts a one-space policy id. GapAnalysisRequest
then rejects it, and that ValidationError is returned as a generic HTTP 500.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_suite_schemas import GapAnalysisResponse, GapSummary
from gap_analysis_service_v4 import GapAnalysisServiceV4Error


def _client():
    app = Flask(__name__)
    app.register_blueprint(compliance_bp)
    return app.test_client()


def _install(config, service):
    previous = (compliance_routes._config, compliance_routes._gap_analysis_v4_service)
    compliance_routes._config = config
    compliance_routes._gap_analysis_v4_service = service
    return previous


def _restore(previous) -> None:
    compliance_routes._config, compliance_routes._gap_analysis_v4_service = previous


def test_route_forwards_v4_status_and_hides_unexpected_errors() -> None:
    config = load_config()
    config.auth_required = False
    config.compliance_database = "privacy-compliance"
    service = MagicMock()
    service.run = AsyncMock(side_effect=GapAnalysisServiceV4Error("Policy not found: pol-1", 404))
    previous = _install(config, service)
    try:
        missing = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA", "policy_collection": "policies"},
        )
        assert missing.status_code == 404
        assert missing.get_json()["error"] == "Policy not found: pol-1"
        forwarded = service.run.call_args[0][0]
        assert forwarded.policy_document_id == "pol-1"
        assert forwarded.applicable_jurisdictions == ["CA"]
        assert forwarded.database == "privacy-compliance"
        assert forwarded.policy_collection == "policies"
        assert forwarded.save_results is False
        assert forwarded.run_async is False

        service.run = AsyncMock(side_effect=GapAnalysisServiceV4Error("Rate limit exceeded", 429))
        limited = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
        )
        assert limited.status_code == 429
        assert limited.get_json()["error"] == "Rate limit exceeded"

        service.run = AsyncMock(side_effect=RuntimeError("mongodb://secret-host/privacy"))
        crashed = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
        )
        assert crashed.status_code == 500
        assert crashed.get_json() == {"error": "Internal server error"}
        assert "secret-host" not in crashed.get_data(as_text=True)
    finally:
        _restore(previous)


def test_blank_policy_id_is_422_and_whitespace_id_is_generic_500() -> None:
    config = load_config()
    config.auth_required = False
    service = MagicMock()
    service.run = AsyncMock()
    previous = _install(config, service)
    try:
        blank = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "", "jurisdiction": "CA"},
        )
        assert blank.status_code == 422
        assert blank.get_json()["error"] == "Validation error"

        spaced = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": " ", "jurisdiction": " CA "},
        )
        assert spaced.status_code == 500
        assert spaced.get_json() == {"error": "Internal server error"}
        service.run.assert_not_called()

        empty_collection = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA", "policy_collection": ""},
        )
        assert empty_collection.status_code == 422
        service.run.assert_not_called()

        service.run = AsyncMock(
            return_value=GapAnalysisResponse(
                policy_document_id="  pol-9  ",
                applicable_jurisdictions=[" CA "],
                analyzed_at="2026-01-01T00:00:00+00:00",
                gaps=[],
                summary=GapSummary(),
            )
        )
        padded = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "  pol-9  ", "jurisdiction": " CA "},
        )
        assert padded.status_code == 200
        body = padded.get_json()
        assert body["policy_id"] == "  pol-9  "
        assert body["applicable_jurisdictions"] == [" CA "]
        assert body["warnings"] == []
        forwarded = service.run.call_args[0][0]
        assert forwarded.policy_document_id == "  pol-9  "
        assert forwarded.applicable_jurisdictions == [" CA "]
        assert forwarded.save_results is False
        assert forwarded.run_async is False
    finally:
        _restore(previous)


def test_auth_and_initialization_failures_stop_before_gap_analysis() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = "secret"
    service = MagicMock()
    service.run = AsyncMock()
    previous = _install(config, service)
    try:
        missing_key = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
        )
        assert missing_key.status_code == 401
        assert missing_key.get_json()["error"] == "Missing API key."

        wrong_key = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
            headers={"x-api-key": "nope"},
        )
        assert wrong_key.status_code == 403
        service.run.assert_not_called()

        compliance_routes._gap_analysis_v4_service = None
        compliance_routes._config = config
        config.auth_required = False
        uninitialized = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
        )
        assert uninitialized.status_code == 500
        assert uninitialized.get_json() == {"error": "Internal server error"}

        compliance_routes._config = None
        no_config = _client().post(
            "/statute-policy-compliance",
            json={"policy_id": "pol-1", "jurisdiction": "CA"},
        )
        assert no_config.status_code == 500
        assert no_config.get_json()["error"] == "Compliance config not initialized."
    finally:
        _restore(previous)
