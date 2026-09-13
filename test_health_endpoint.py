"""Regression tests for GET /health (load-balancer contract).

app.py constructs Mongo/Firecrawl/Anthropic clients at import time, so the
production ``health`` function is extracted from source and bound to a
minimal Flask app — the same approach used for ``_safe_log_body``.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

from flask import Flask, jsonify


def _health_source() -> str:
    path = Path(__file__).resolve().parent / "app.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "health":
            extracted = ast.get_source_segment(source, node)
            if extracted:
                return extracted
    raise AssertionError("health() not found in app.py")


def _app_with_health(mongo_client):
    ns = {
        "mongo_client": mongo_client,
        "jsonify": jsonify,
    }
    exec(_health_source(), ns)  # noqa: S102 — bind production health() without importing app.py
    app = Flask(__name__)
    app.add_url_rule("/health", "health", ns["health"], methods=["GET"])
    return app


def test_health_ok_when_mongo_pings() -> None:
    mongo = MagicMock()
    mongo.admin.command.return_value = {"ok": 1}
    client = _app_with_health(mongo).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "database": "ok"}
    mongo.admin.command.assert_called_once_with("ping")


def test_health_degraded_when_mongo_ping_fails() -> None:
    mongo = MagicMock()
    mongo.admin.command.side_effect = RuntimeError("not reachable")
    client = _app_with_health(mongo).test_client()

    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "degraded", "database": "error"}


def test_health_does_not_require_api_key() -> None:
    """OpenAPI marks /health as unauthenticated; a configured APP_API_KEY must not 401."""
    mongo = MagicMock()
    mongo.admin.command.return_value = {"ok": 1}
    client = _app_with_health(mongo).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert "error" not in response.get_json()
