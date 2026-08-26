"""Regression tests for POST /statute-policy-compliance request mapping.

GapAnalysisRequest.run_async defaults to True (background job). The legacy
statute-policy endpoint must force a synchronous v4 run and surface v4 errors.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_suite_schemas import GapAnalysisResponse, GapSummary, RetrievalMetadata
from gap_analysis_service_v4 import GapAnalysisServiceV4Error


def _ensure_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _post(payload: dict, mock_v4: Any):
    """POST /statute-policy-compliance with auth disabled and a mocked v4 service."""
    _ensure_event_loop()
    config = load_config()
    config.auth_required = False
    old_v4 = compliance_routes._gap_analysis_v4_service
    old_config = compliance_routes._config
    compliance_routes._gap_analysis_v4_service = mock_v4
    compliance_routes._config = config
    try:
        app = Flask(__name__)
        app.register_blueprint(compliance_bp)
        client = app.test_client()
        return client.post("/statute-policy-compliance", json=payload)
    finally:
        compliance_routes._gap_analysis_v4_service = old_v4
        compliance_routes._config = old_config


def test_statute_policy_forces_synchronous_v4_run() -> None:
    """Omitting run_async must not inherit GapAnalysisRequest's default True."""
    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="pol-1",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-08-26T00:00:00Z",
            gaps=[],
            summary=GapSummary(),
            retrieval_metadata=RetrievalMetadata(
                statute_items_considered=0, statute_pairs_matched=0
            ),
        )
    )
    response = _post({"policy_id": "pol-1", "jurisdiction": "california"}, mock_v4)
    assert response.status_code == 200
    req = mock_v4.run.call_args[0][0]
    assert req.policy_document_id == "pol-1"
    assert req.applicable_jurisdictions == ["california"]
    assert req.save_results is False
    assert req.run_async is False
    mock_v4.run.assert_awaited_once()


def test_statute_policy_maps_v4_error_status() -> None:
    """GapAnalysisServiceV4Error status_code is returned, not a generic 500."""
    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock(
        side_effect=GapAnalysisServiceV4Error(
            "Policy not found: pol-missing", status_code=404
        )
    )
    response = _post({"policy_id": "pol-missing", "jurisdiction": "CA"}, mock_v4)
    assert response.status_code == 404
    assert "pol-missing" in response.get_json()["error"]


def test_statute_policy_validation_error_is_422() -> None:
    """Missing required policy_id is a 422 validation error, not a v4 call."""
    mock_v4 = MagicMock()
    mock_v4.run = AsyncMock()
    response = _post({"jurisdiction": "CA"}, mock_v4)
    assert response.status_code == 422
    body = response.get_json()
    assert body["error"] == "Validation error"
    mock_v4.run.assert_not_called()
