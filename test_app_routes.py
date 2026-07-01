"""Tests for app root, 404 handler, and routes blueprint (openapi.json, docs)."""
from __future__ import annotations

import pytest
from flask import Flask


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


def test_openapi_documents_adaptive_feedback_endpoints(client_routes) -> None:
    """Adaptive-feedback endpoints must stay represented in OpenAPI."""
    response = client_routes.get("/openapi.json")
    assert response.status_code == 200
    paths = response.get_json()["paths"]

    feedback_path = paths["/api/v4/compliance/adaptive-feedback"]["get"]
    assert feedback_path["summary"] == "List adaptive feedback history"
    assert any(
        param["name"] == "policy_document_id" and param.get("required") is True
        for param in feedback_path["parameters"]
    )
    assert "/api/v4/compliance/adaptive-feedback/log" in paths


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
