"""Request-log channels that bypass `_safe_log_body`.

Loaded from app.py source so the suite does not boot Mongo, Firecrawl, or
Anthropic (app.py connects to those clients at import).

These cases are the channels left open after top-level JSON redaction:
hyphenated key names, nested values, form bodies, and query strings.
"""
from __future__ import annotations

import ast
import copy
import json
import logging
from pathlib import Path

import pytest
from flask import Flask, request


_APP_PATH = Path(__file__).resolve().parent / "app.py"
_HELPER_NAMES = {"_safe_log_body", "log_request_params"}


def _load_helpers(logger: logging.Logger) -> dict:
    selected = []
    tree = ast.parse(_APP_PATH.read_text())
    for node in tree.body:
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
    namespace = {"json": json, "request": request, "logger": logger}
    exec(compile(module, str(_APP_PATH), "exec"), namespace)
    return namespace


@pytest.fixture
def log_helpers():
    logger = logging.getLogger("web-gather-api.unredacted-channels")
    logger.handlers.clear()
    logger.propagate = True
    return _load_helpers(logger)


def test_hyphenated_and_nested_secrets_skip_the_key_scan(log_helpers) -> None:
    """Redaction is a top-level substring check, not a recursive walk.

    `api-key` and `x-api-key` do not contain `api_key` or `apikey`. A nested
    `password` under a safe key, and a list body, are copied through unchanged.
    A matching sibling key is still replaced.
    """
    safe_log_body = log_helpers["_safe_log_body"]
    hyphen_value = "hyphen-api-key-value"
    header_style_value = "header-style-api-key-value"
    nested_value = "nested-password-value"
    list_value = "list-password-value"
    body = {
        "api_key": "underscore-key-value",
        "api-key": hyphen_value,
        "x-api-key": header_style_value,
        "credentials": {"password": nested_value, "note": "keep-nested-note"},
        "items": [{"password": list_value}],
    }

    redacted = safe_log_body(body)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["api-key"] == hyphen_value
    assert redacted["x-api-key"] == header_style_value
    assert redacted["credentials"]["password"] == nested_value
    assert redacted["credentials"]["note"] == "keep-nested-note"
    assert redacted["items"][0]["password"] == list_value
    serialized = json.dumps(redacted)
    assert "underscore-key-value" not in serialized
    assert hyphen_value in serialized
    assert header_style_value in serialized
    assert nested_value in serialized
    assert list_value in serialized


def test_list_body_is_not_scanned_for_sensitive_keys(log_helpers) -> None:
    safe_log_body = log_helpers["_safe_log_body"]
    body = [{"password": "list-root-password", "api-key": "list-root-key"}]
    assert safe_log_body(body) == body


def test_patch_json_logs_hyphenated_key_and_still_redacts_password(log_helpers, caplog) -> None:
    """PATCH is in the JSON logging set. Hyphenated keys stay in the log line."""
    log_request_params = log_helpers["log_request_params"]
    app = Flask(__name__)
    app.before_request(log_request_params)

    @app.route("/hook", methods=["PATCH"])
    def hook():
        return {"ok": True}

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger=log_helpers["logger"].name):
        response = client.patch(
            "/hook",
            json={
                "password": "patch-password-value",
                "x-api-key": "patch-hyphen-key-value",
                "policy": "retain 30 days",
            },
        )
    assert response.status_code == 200
    logged = " ".join(caplog.messages)
    assert "patch-password-value" not in logged
    assert "[REDACTED]" in logged
    assert "patch-hyphen-key-value" in logged
    assert "retain 30 days" in logged
    assert '"body"' in logged


def test_form_and_query_credentials_are_logged_verbatim(log_helpers, caplog) -> None:
    """Form bodies and query strings never pass through `_safe_log_body`."""
    log_request_params = log_helpers["log_request_params"]
    app = Flask(__name__)
    app.before_request(log_request_params)

    @app.route("/hook", methods=["GET", "POST"])
    def hook():
        return {"ok": True}

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger=log_helpers["logger"].name):
        form_response = client.post(
            "/hook",
            data={"password": "form-password-value", "company": "Acme"},
        )
        query_response = client.get("/hook?api_key=query-api-key-value&q=privacy")
    assert form_response.status_code == 200
    assert query_response.status_code == 200

    form_log = caplog.messages[0]
    query_log = caplog.messages[1]
    assert '"form"' in form_log
    assert "form-password-value" in form_log
    assert "Acme" in form_log
    assert "[REDACTED]" not in form_log
    assert '"query"' in query_log
    assert "query-api-key-value" in query_log
    assert "privacy" in query_log
    assert "[REDACTED]" not in query_log
    assert '"body"' not in form_log
    assert '"body"' not in query_log
