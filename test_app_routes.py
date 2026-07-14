"""Tests for app root, 404 handler, and routes blueprint (openapi.json, docs)."""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask


@pytest.fixture
def app_module(monkeypatch):
    """Import the real app with external clients replaced by deterministic mocks."""
    import anthropic
    import firecrawl
    import pymongo

    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    mongo_client = MagicMock()
    monkeypatch.setattr(pymongo, "MongoClient", MagicMock(return_value=mongo_client))
    monkeypatch.setattr(firecrawl, "FirecrawlApp", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(anthropic, "Anthropic", MagicMock(return_value=MagicMock()))

    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    yield module
    sys.modules.pop("app", None)


@pytest.fixture
def app_with_routes():
    """Flask app with routes blueprint only (no env vars required)."""
    from routes import routes_bp

    app = Flask(__name__)
    app.register_blueprint(routes_bp)
    return app


@pytest.fixture
def client_routes(app_with_routes):
    """Test client for routes blueprint."""
    return app_with_routes.test_client()


def test_openapi_json_returns_valid_spec(client_routes) -> None:
    """GET /openapi.json returns valid OpenAPI spec."""
    response = client_routes.get("/openapi.json")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.3"
    assert data["info"]["title"] == "Web Gather API"
    assert "paths" in data
    assert "components" in data


def test_docs_returns_html(client_routes) -> None:
    """GET /docs returns 200 with HTML content."""
    response = client_routes.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.content_type
    assert b"swagger-ui" in response.data
    assert b"openapi.json" in response.data


def test_docs_trailing_redirects_to_docs(client_routes) -> None:
    """GET /docs/ redirects to /docs."""
    response = client_routes.get("/docs/")
    assert response.status_code == 302
    assert response.location and "/docs" in response.location


@pytest.fixture
def app_with_root_and_404():
    """Minimal app mimicking app.py root and 404 behavior (no env vars)."""
    from flask import redirect, request
    from routes import routes_bp

    app = Flask(__name__)
    app.register_blueprint(routes_bp)

    @app.get("/")
    def index():
        return redirect("/docs", code=302)

    @app.errorhandler(404)
    def redirect_404_to_docs(_exc):
        if request.method == "GET" and not request.path.startswith("/api"):
            return redirect("/docs", code=302)
        from flask import jsonify
        return jsonify({"error": "Not found"}), 404

    return app


@pytest.fixture
def client_full(app_with_root_and_404):
    """Test client for app with root and 404 handlers."""
    return app_with_root_and_404.test_client()


def test_root_redirects_to_docs(client_full) -> None:
    """GET / redirects to /docs."""
    response = client_full.get("/")
    assert response.status_code == 302
    assert response.location and "/docs" in response.location


def test_404_non_api_get_redirects_to_docs(client_full) -> None:
    """GET for non-API path (e.g. /unknown-page) redirects to /docs."""
    response = client_full.get("/unknown-page")
    assert response.status_code == 302
    assert response.location and "/docs" in response.location


def test_404_api_returns_json(client_full) -> None:
    """GET /api/nonexistent returns 404 JSON (API paths start with /api)."""
    response = client_full.get("/api/nonexistent")
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data


def test_health_reports_ok_when_database_ping_succeeds(app_module) -> None:
    """A successful MongoDB ping keeps the service ready for traffic."""
    app_module.mongo_client.admin.command.return_value = {"ok": 1}

    response = app_module.app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "database": "ok"}
    app_module.mongo_client.admin.command.assert_called_once_with("ping")


def test_health_reports_degraded_when_database_ping_fails(app_module) -> None:
    """Database failure must make the load-balancer health check fail closed."""
    app_module.mongo_client.admin.command.side_effect = RuntimeError("database unavailable")

    response = app_module.app.test_client().get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "degraded", "database": "error"}


def test_safe_log_body_redacts_secrets_and_truncates_large_payloads(app_module) -> None:
    """Request logging must not expose credentials or unbounded payloads."""
    redacted = app_module._safe_log_body(
        {
            "username": "alice",
            "Password": "do-not-log",
            "authorization_token": "also-secret",
        }
    )

    assert redacted == {
        "username": "alice",
        "Password": "[REDACTED]",
        "authorization_token": "[REDACTED]",
    }

    truncated = app_module._safe_log_body({"content": "x" * 1000})
    assert isinstance(truncated, str)
    assert truncated.endswith("... [truncated]")
    assert len(truncated) == app_module._LOG_BODY_MAX_LEN + len("... [truncated]")
