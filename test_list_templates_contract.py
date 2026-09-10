"""Risk-assessment templates service and uninitialized-config route contracts."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_routes import compliance_bp
from compliance_suite_service import ComplianceSuiteService
import compliance_routes


def test_list_templates_returns_default_dpia() -> None:
    """Service always returns the hardcoded default DPIA-style template."""
    svc = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=load_config(),
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=MagicMock(),
    )
    result = asyncio.run(svc.list_templates())
    dumped = result.model_dump()
    assert dumped == {
        "templates": [{"id": "default", "label": "Default DPIA-style"}],
    }


def test_templates_uninitialized_config_is_json_500() -> None:
    """GET /risk-assessment/templates maps _get_config RuntimeError to JSON 500."""
    config = load_config()
    config.auth_required = False
    app = Flask(__name__)
    app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
    with patch.object(
        compliance_routes,
        "_get_config",
        side_effect=RuntimeError("Compliance config not initialized"),
    ):
        client = app.test_client()
        response = client.get("/api/compliance/risk-assessment/templates")
    assert response.status_code == 500
    assert response.get_json()["error"] == "Compliance config not initialized"
