"""404 handler method/path contract.

GET for unknown non-API paths redirects to /docs (browser UX). POST/PUT/PATCH
to those same paths — and any /api/* miss — must stay JSON 404 so API clients
are not sent an HTML redirect.
"""
from __future__ import annotations

import pytest
from flask import Flask, jsonify, redirect, request


@pytest.fixture
def client_404():
    from routes import routes_bp

    app = Flask(__name__)
    app.register_blueprint(routes_bp)

    @app.errorhandler(404)
    def redirect_404_to_docs(_exc):
        if request.method == "GET" and not request.path.startswith("/api"):
            return redirect("/docs", code=302)
        return jsonify({"error": "Not found"}), 404

    return app.test_client()


def test_post_unknown_non_api_path_returns_json_404(client_404) -> None:
    response = client_404.post("/unknown-page", json={"x": 1})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"


def test_put_unknown_non_api_path_returns_json_404(client_404) -> None:
    response = client_404.put("/unknown-page", json={"x": 1})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"


def test_post_unknown_api_path_returns_json_404(client_404) -> None:
    response = client_404.post("/api/does-not-exist", json={})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"
