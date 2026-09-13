"""Regression tests for the unused-but-shared async APP_API_KEY decorator."""
from __future__ import annotations

import asyncio

from flask import Flask, jsonify

import security
from security import require_api_key_async


def _restore_key(previous) -> None:
    security._app_api_key = previous


def test_require_api_key_async_missing_header_is_401() -> None:
    previous = security._app_api_key
    security._app_api_key = "secret"
    app = Flask(__name__)

    @require_api_key_async
    async def view():
        return jsonify({"ok": True}), 200

    try:
        with app.test_request_context("/x", headers={}):
            resp, code = asyncio.run(view())
        assert code == 401
        assert resp.get_json()["error"] == "Missing API key."
    finally:
        _restore_key(previous)


def test_require_api_key_async_whitespace_header_is_401() -> None:
    previous = security._app_api_key
    security._app_api_key = "secret"
    app = Flask(__name__)

    @require_api_key_async
    async def view():
        return jsonify({"ok": True}), 200

    try:
        with app.test_request_context("/x", headers={"x-api-key": "   "}):
            resp, code = asyncio.run(view())
        assert code == 401
        assert resp.get_json()["error"] == "Missing API key."
    finally:
        _restore_key(previous)


def test_require_api_key_async_mismatch_is_403() -> None:
    previous = security._app_api_key
    security._app_api_key = "secret"
    app = Flask(__name__)

    @require_api_key_async
    async def view():
        return jsonify({"ok": True}), 200

    try:
        with app.test_request_context("/x", headers={"x-api-key": "wrong"}):
            resp, code = asyncio.run(view())
        assert code == 403
        assert resp.get_json()["error"] == "Invalid API key."
    finally:
        _restore_key(previous)


def test_require_api_key_async_strips_matching_header() -> None:
    previous = security._app_api_key
    security._app_api_key = "secret"
    app = Flask(__name__)

    @require_api_key_async
    async def view():
        return jsonify({"ok": True}), 200

    try:
        with app.test_request_context("/x", headers={"x-api-key": "  secret  "}):
            resp, code = asyncio.run(view())
        assert code == 200
        assert resp.get_json() == {"ok": True}
    finally:
        _restore_key(previous)


def test_require_api_key_async_disabled_when_key_unset() -> None:
    previous = security._app_api_key
    security._app_api_key = None
    app = Flask(__name__)

    @require_api_key_async
    async def view():
        return jsonify({"ok": True}), 200

    try:
        with app.test_request_context("/x", headers={}):
            resp, code = asyncio.run(view())
        assert code == 200
        assert resp.get_json() == {"ok": True}
    finally:
        _restore_key(previous)
