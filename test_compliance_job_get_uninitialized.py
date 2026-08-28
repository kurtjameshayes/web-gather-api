"""GET /api/compliance/jobs/<id> uninitialized-storage vs uninitialized-config.

Auth/config RuntimeError is caught and returned as JSON 500. Job storage is
read *after* that try/except, so an uninitialized `_job_storage` currently
becomes Flask's generic HTTP 500 (no `{"error": ...}` body). Pin both so a
future guard change is intentional.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from flask import Flask

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_routes import compliance_bp
import compliance_routes


def test_get_job_uninitialized_config_returns_json_500() -> None:
    """Missing compliance config is caught and returned as JSON 500."""
    with patch.object(compliance_routes, "_config", None), patch.object(
        compliance_routes, "_job_storage", None
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        client = app.test_client()
        response = client.get("/api/compliance/jobs/job-1")

    assert response.status_code == 500
    payload = response.get_json()
    assert payload is not None
    assert payload["error"] == "Compliance config not initialized."


def test_get_job_uninitialized_storage_is_generic_500() -> None:
    """Uninitialized job storage after auth is not caught (generic HTTP 500)."""
    config = load_config()
    config.auth_required = False
    with patch.object(compliance_routes, "_config", config), patch.object(
        compliance_routes, "_job_storage", None
    ):
        app = Flask(__name__)
        app.register_blueprint(compliance_bp, url_prefix="/api/compliance")
        client = app.test_client()
        response = client.get("/api/compliance/jobs/job-1")

    assert response.status_code == 500
    # Flask's default handler does not JSON-encode an uncaught RuntimeError.
    payload = response.get_json(silent=True)
    assert payload is None or "Compliance job storage not initialized" not in str(payload)
