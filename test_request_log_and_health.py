"""Request-log redaction and /health status mapping.

Loaded from app.py source so the suite does not boot Mongo, Firecrawl, or
Anthropic (app.py connects to those clients at import).
"""
from __future__ import annotations

import ast
import copy
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from flask import Flask, jsonify, request


_APP_PATH = Path(__file__).resolve().parent / "app.py"
_HELPER_NAMES = {"_safe_log_body", "log_request_params", "health"}


def _app_tree() -> ast.Module:
    return ast.parse(_APP_PATH.read_text())


def _load_helpers(mongo_client: object, logger: logging.Logger) -> dict:
    selected = []
    for node in _app_tree().body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "_LOG_BODY_MAX_LEN" in names:
                selected.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES:
            fn = copy.deepcopy(node)
            fn.decorator_list = []
            selected.append(fn)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "request": request,
        "jsonify": jsonify,
        "logger": logger,
        "mongo_client": mongo_client,
    }
    exec(compile(module, str(_APP_PATH), "exec"), namespace)
    return namespace


@pytest.fixture
def log_helpers():
    logger = logging.getLogger("web-gather-api.request-log-test")
    logger.handlers.clear()
    logger.propagate = True
    return _load_helpers(MagicMock(), logger)


def test_log_hook_is_a_before_request() -> None:
    """The redacting logger must stay on the application before-request hook."""
    fn = next(
        node
        for node in _app_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "log_request_params"
    )
    assert any(
        isinstance(dec, ast.Attribute) and dec.attr == "before_request"
        for dec in fn.decorator_list
    )


def test_safe_log_body_redacts_sensitive_keys_case_and_substring(log_helpers) -> None:
    safe_log_body = log_helpers["_safe_log_body"]
    body = {
        "password": "hunter2",
        "API_KEY": "top-secret",
        "access_token": "tok-1",
        "client_secret": "sec-1",
        "authorization": "Bearer abc",
        "apikey": "raw-key",
        "note": "keep-me",
    }
    redacted = safe_log_body(body)
    assert redacted["note"] == "keep-me"
    for key in ("password", "API_KEY", "access_token", "client_secret", "authorization", "apikey"):
        assert redacted[key] == "[REDACTED]"
    serialized = json.dumps(redacted)
    for secret in ("hunter2", "top-secret", "tok-1", "sec-1", "Bearer abc", "raw-key"):
        assert secret not in serialized


def test_safe_log_body_redacts_before_truncation(log_helpers) -> None:
    """A long secret must not survive into the truncated log prefix."""
    safe_log_body = log_helpers["_safe_log_body"]
    secret = "SUPERSECRET" + ("x" * 40)
    body = {"password": secret, "note": "n" * 600}
    unredacted = json.dumps(body)
    assert secret in unredacted[:500]

    logged = safe_log_body(body)
    assert isinstance(logged, str)
    assert logged.endswith("... [truncated]")
    assert secret not in logged
    assert "[REDACTED]" in logged


def test_safe_log_body_truncation_boundary(log_helpers) -> None:
    safe_log_body = log_helpers["_safe_log_body"]
    limit = log_helpers["_LOG_BODY_MAX_LEN"]
    assert limit == 500
    inner = limit - len('{"n": "') - len('"}')
    exact = {"n": "a" * inner}
    assert len(json.dumps(exact)) == limit
    assert safe_log_body(exact) == exact

    over = {"n": "a" * (inner + 1)}
    logged = safe_log_body(over)
    assert logged == json.dumps(over)[:limit] + "... [truncated]"


def test_safe_log_body_none_and_non_objects(log_helpers) -> None:
    safe_log_body = log_helpers["_safe_log_body"]
    assert safe_log_body(None) is None
    assert safe_log_body(["keep", 1]) == ["keep", 1]
    assert safe_log_body("plain") == "plain"
    assert safe_log_body({}) == {}

    class Weird:
        def __str__(self) -> str:
            return "w" * 600

    logged = safe_log_body(Weird())
    assert isinstance(logged, str)
    assert logged.endswith("... [truncated]")
    assert logged.startswith('"')


def test_request_logger_redacts_json_body_and_keeps_query(log_helpers, caplog) -> None:
    log_request_params = log_helpers["log_request_params"]
    app = Flask(__name__)
    app.before_request(log_request_params)

    @app.route("/hook", methods=["GET", "POST"])
    def hook():
        return {"ok": True}

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger=log_helpers["logger"].name):
        response = client.post(
            "/hook?q=privacy",
            json={"password": "hunter2", "policy": "retain 30 days"},
        )
    assert response.status_code == 200
    logged = " ".join(caplog.messages)
    assert "hunter2" not in logged
    assert "[REDACTED]" in logged
    assert "retain 30 days" in logged
    assert "privacy" in logged


def test_request_logger_omits_null_json_and_logs_form(log_helpers, caplog) -> None:
    log_request_params = log_helpers["log_request_params"]
    app = Flask(__name__)
    app.before_request(log_request_params)

    @app.route("/hook", methods=["POST"])
    def hook():
        return {"ok": True}

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger=log_helpers["logger"].name):
        null_response = client.post("/hook", data="null", content_type="application/json")
        form_response = client.post("/hook", data={"company": "Acme"})
    assert null_response.status_code == 200
    assert form_response.status_code == 200
    null_log = caplog.messages[0]
    form_log = caplog.messages[1]
    assert '"body"' not in null_log
    assert "Acme" in form_log
    assert '"form"' in form_log


def test_health_ok_and_degraded() -> None:
    logger = logging.getLogger("web-gather-api.health-test")
    app = Flask(__name__)

    mongo_ok = MagicMock()
    helpers = _load_helpers(mongo_ok, logger)
    with app.app_context():
        ok_body, ok_code = helpers["health"]()
    assert ok_code == 200
    assert ok_body.get_json() == {"status": "ok", "database": "ok"}
    mongo_ok.admin.command.assert_called_once_with("ping")

    mongo_down = MagicMock()
    mongo_down.admin.command.side_effect = RuntimeError("connection refused")
    helpers = _load_helpers(mongo_down, logger)
    with app.app_context():
        bad_body, bad_code = helpers["health"]()
    assert bad_code == 503
    assert bad_body.get_json() == {"status": "degraded", "database": "error"}
