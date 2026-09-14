"""POST /consumer-rights-router stringifies Pydantic error ctx.

Other compliance POST handlers return exc.errors() unchanged. This route
converts `ctx` values to str so jsonify cannot fail on non-JSON ctx objects.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_config import load_config
from compliance_routes import compliance_bp


@pytest.fixture
def client():
    config = load_config()
    config.auth_required = False
    suite = MagicMock()
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_get_suite_service", return_value=suite
    ), patch.object(compliance_routes, "_get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        yield app.test_client(), suite


def test_consumer_rights_router_missing_source_is_422_json(client) -> None:
    http, suite = client
    response = http.post("/api/compliance/consumer-rights-router", json={})
    assert response.status_code == 422
    data = response.get_json()
    assert data["error"] == "Validation error"
    assert isinstance(data["details"], list)
    assert data["details"]
    for err in data["details"]:
        if "ctx" in err:
            assert isinstance(err["ctx"], str)
    suite.consumer_rights_router.assert_not_called()


def test_consumer_rights_router_invalid_request_types_stringifies_ctx(client) -> None:
    http, suite = client
    response = http.post(
        "/api/compliance/consumer-rights-router",
        json={"text": "delete my data", "request_types": ["not-a-real-type"]},
    )
    assert response.status_code == 422
    data = response.get_json()
    assert data["error"] == "Validation error"
    blob = str(data["details"]).lower()
    assert "request_types" in blob or "invalid" in blob
    for err in data["details"]:
        if "ctx" in err:
            assert isinstance(err["ctx"], str)
    suite.consumer_rights_router.assert_not_called()
