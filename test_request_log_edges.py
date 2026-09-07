"""Leftover request-log redaction contracts not covered on base or PR #180.

PR #180 pins top-level Password/authorization_token redaction and truncation
of oversized JSON bodies. These cases pin nested/list/None behavior, substring
key matching, and query/form logging that currently bypasses redaction.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_log_helpers() -> dict[str, Any]:
    """Load logging helpers from app.py without constructing Mongo/Firecrawl clients."""
    src = Path(__file__).resolve().parent.joinpath("app.py").read_text(encoding="utf-8")
    start = src.index("_LOG_BODY_MAX_LEN")
    end = src.index("if __name__")
    chunk = src[start:end].replace("@app.before_request\n", "")
    ns: dict[str, Any] = {"json": json, "logger": logging.getLogger("web-gather-api")}
    exec(chunk, ns)
    return ns


_HELPERS = _load_log_helpers()
_safe_log_body = _HELPERS["_safe_log_body"]


def test_safe_log_body_does_not_redact_nested_secrets() -> None:
    """Only top-level keys are scanned; nested api_key values currently leak."""
    redacted = _safe_log_body(
        {"user": {"api_key": "nested-secret", "name": "alice"}, "ok": True}
    )
    assert redacted["ok"] is True
    assert redacted["user"]["api_key"] == "nested-secret"
    assert redacted["user"]["name"] == "alice"


def test_safe_log_body_redacts_substring_sensitive_keys() -> None:
    """Matching is substring-based: token_count / secret_flag are redacted today."""
    redacted = _safe_log_body(
        {"token_count": 3, "secret_flag": True, "note": "visible"}
    )
    assert redacted["token_count"] == "[REDACTED]"
    assert redacted["secret_flag"] == "[REDACTED]"
    assert redacted["note"] == "visible"


def test_safe_log_body_passes_through_none_and_list_bodies() -> None:
    """None stays None. JSON arrays are logged without key redaction."""
    assert _safe_log_body(None) is None
    listed = _safe_log_body(["api_key", "password", {"token": "x"}])
    assert listed == ["api_key", "password", {"token": "x"}]


def test_log_request_params_logs_query_string_unredacted(caplog) -> None:
    """GET query params are copied into the log without _safe_log_body."""
    request = SimpleNamespace(
        method="GET",
        path="/health",
        args={"api_key": "query-secret", "q": "ok"},
        is_json=False,
        form={},
    )
    log_fn = _HELPERS["log_request_params"]
    log_fn.__globals__["request"] = request

    with caplog.at_level(logging.INFO, logger="web-gather-api"):
        log_fn()

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "query-secret" in logged
    assert "/health" in logged


def test_log_request_params_logs_form_fields_unredacted(caplog) -> None:
    """POST form bodies skip JSON redaction and log raw field values."""
    request = SimpleNamespace(
        method="POST",
        path="/ingest",
        args={},
        is_json=False,
        form={"password": "form-secret", "url": "https://example.com"},
    )
    log_fn = _HELPERS["log_request_params"]
    log_fn.__globals__["request"] = request

    with caplog.at_level(logging.INFO, logger="web-gather-api"):
        log_fn()

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "form-secret" in logged
    assert "https://example.com" in logged
