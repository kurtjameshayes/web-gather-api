"""v2/v3/v4 gap-analysis validation jsonify vs consumer-rights ctx stringify.

POST /consumer-rights-router stringifies Pydantic ctx (#235) so jsonify cannot
fail. These handlers pass exc.errors() unchanged:

- A model-validator ValueError (missing policy id) puts a ValueError in ctx,
  so jsonify raises and Flask returns a generic HTTP 500.
- Field constraints with JSON-safe ctx (num_rows=0) still return 422 with
  raw dict ctx, and the service is not called.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import compliance_routes
import compliance_routes_v2
import compliance_routes_v3
import compliance_routes_v4
from compliance_config import load_config
from compliance_routes_v2 import compliance_v2_bp
from compliance_routes_v3 import compliance_v3_bp
from compliance_routes_v4 import compliance_v4_bp


def _v2_client(mock_suite: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes_v2, "_get_suite_service", return_value=mock_suite
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_v2_bp, url_prefix="/api/v2/compliance")
        return app.test_client()


def _v3_client(mock_v3: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch(
        "compliance_routes_v3._get_gap_analysis_v3_service", return_value=mock_v3
    ), patch("compliance_routes_v3._get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_v3_bp, url_prefix="/api/v3/compliance")
        return app.test_client()


def _v4_client(mock_v4: MagicMock):
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch(
        "compliance_routes_v4._get_gap_analysis_v4_service", return_value=mock_v4
    ), patch("compliance_routes_v4._get_job_storage", return_value=MagicMock()):
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        return app.test_client()


def _assert_validator_500(response, mock_service_method) -> None:
    """Missing policy source currently cannot jsonify ValueError ctx."""
    assert response.status_code == 500
    payload = response.get_json(silent=True)
    assert payload is None or payload.get("error") != "Validation error"
    mock_service_method.assert_not_called()


def _assert_field_422(response, mock_service_method) -> None:
    assert response.status_code == 422
    data = response.get_json()
    assert data["error"] == "Validation error"
    assert isinstance(data["details"], list)
    assert data["details"]
    blob = str(data["details"]).lower()
    assert "num_rows" in blob or "greater" in blob
    for err in data["details"]:
        if "ctx" in err:
            assert isinstance(err["ctx"], dict)
    mock_service_method.assert_not_called()


def test_v2_gap_analysis_validator_500_and_field_422() -> None:
    mock_suite = MagicMock()
    client = _v2_client(mock_suite)
    _assert_validator_500(
        client.post("/api/v2/compliance/gap-analysis", json={}),
        mock_suite.gap_analysis_v2,
    )
    _assert_field_422(
        client.post(
            "/api/v2/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "num_rows": 0},
        ),
        mock_suite.gap_analysis_v2,
    )


def test_v3_gap_analysis_validator_500_and_field_422() -> None:
    mock_v3 = MagicMock()
    client = _v3_client(mock_v3)
    _assert_validator_500(
        client.post("/api/v3/compliance/gap-analysis", json={}),
        mock_v3.run,
    )
    _assert_field_422(
        client.post(
            "/api/v3/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "num_rows": 0},
        ),
        mock_v3.run,
    )


def test_v4_gap_analysis_validator_500_and_field_422() -> None:
    mock_v4 = MagicMock()
    client = _v4_client(mock_v4)
    _assert_validator_500(
        client.post("/api/v4/compliance/gap-analysis", json={}),
        mock_v4.run,
    )
    _assert_field_422(
        client.post(
            "/api/v4/compliance/gap-analysis",
            json={"policy_document_id": "pol-1", "num_rows": 0},
        ),
        mock_v4.run,
    )
