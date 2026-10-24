"""Blank API-key configuration for core routes and compliance authorization.

APP_API_KEY is stripped at startup. A whitespace-only value leaves core routes
open. A padded value is stored without the padding. A compliance config key
that is only spaces is truthy and is not stripped, so no request key can match
it. An empty config key skips the comparison and still enforces roles.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from flask import Flask, jsonify

import security
from compliance_config import load_config
from security import AuthorizationError, authorize_request, init_app_api_key, require_api_key


def _client():
    app = Flask(__name__)

    @app.get("/ping")
    @require_api_key
    def ping():
        return jsonify({"ok": True})

    return app.test_client()


def _restore_api_key(previous_env, previous_global) -> None:
    if previous_env is None:
        os.environ.pop("APP_API_KEY", None)
    else:
        os.environ["APP_API_KEY"] = previous_env
    security._app_api_key = previous_global


def test_whitespace_app_api_key_disables_auth_and_padded_key_is_stripped() -> None:
    previous_env = os.environ.get("APP_API_KEY")
    previous_global = security._app_api_key
    try:
        os.environ["APP_API_KEY"] = "   \t"
        init_app_api_key()
        assert security._app_api_key is None
        open_response = _client().get("/ping")
        assert open_response.status_code == 200
        assert open_response.get_json() == {"ok": True}

        os.environ["APP_API_KEY"] = "  secret  "
        init_app_api_key()
        assert security._app_api_key == "secret"
        client = _client()
        missing = client.get("/ping")
        wrong = client.get("/ping", headers={"x-api-key": "nope"})
        exact = client.get("/ping", headers={"x-api-key": "secret"})
        padded = client.get("/ping", headers={"x-api-key": "  secret  "})
    finally:
        _restore_api_key(previous_env, previous_global)

    assert missing.status_code == 401
    assert missing.get_json()["error"] == "Missing API key."
    assert wrong.status_code == 403
    assert wrong.get_json()["error"] == "Invalid API key."
    assert exact.status_code == 200
    assert padded.status_code == 200


def test_blank_compliance_api_key_skips_compare_and_whitespace_key_matches_nobody() -> None:
    config = load_config()
    config.auth_required = True
    config.api_key = ""
    config.allowed_roles = ["admin"]
    config.default_role_header = "x-role"

    authorize_request(
        config,
        SimpleNamespace(headers={"x-api-key": "not-configured", "x-role": "admin"}),
    )
    with pytest.raises(AuthorizationError) as blocked:
        authorize_request(
            config,
            SimpleNamespace(headers={"x-api-key": "not-configured", "x-role": "viewer"}),
        )
    assert blocked.value.status_code == 403
    assert blocked.value.message == "Insufficient role."

    with pytest.raises(AuthorizationError) as missing:
        authorize_request(config, SimpleNamespace(headers={"x-api-key": "   ", "x-role": "admin"}))
    assert missing.value.status_code == 401

    config.api_key = "  "
    config.allowed_roles = []
    with pytest.raises(AuthorizationError) as blank_header:
        authorize_request(config, SimpleNamespace(headers={"x-api-key": "  "}))
    assert blank_header.value.status_code == 401
    with pytest.raises(AuthorizationError) as unmatched:
        authorize_request(config, SimpleNamespace(headers={"x-api-key": "secret"}))
    assert unmatched.value.status_code == 403
    assert unmatched.value.message == "Invalid API key."
