"""APP_API_KEY gate for core, db, and util routes.

When the key is unset, those routes stay open. When it is set, a missing or
blank key is rejected before the handler runs, and only the x-api-key header
is accepted.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask, jsonify

import security
from core import core_bp, init_core
from db import db_bp, init_db
from security import init_app_api_key, require_api_key_async
from util import init_util, util_bp


@pytest.fixture
def restore_app_api_key():
    previous = security._app_api_key
    yield
    security._app_api_key = previous


@pytest.fixture
def gated(restore_app_api_key) -> dict[str, Any]:
    util_coll = MagicMock()
    util_coll.find.return_value = [{"database_name": "privacy", "model_name": "m"}]
    util_db = MagicMock()
    util_db.__getitem__.return_value = util_coll
    util_mongo = MagicMock()
    util_mongo.__getitem__.return_value = util_db
    init_util(util_mongo)

    user_coll = MagicMock()
    user_coll.delete_many.return_value = MagicMock(deleted_count=2)
    user_db = MagicMock()
    user_db.__getitem__.return_value = user_coll
    db_mongo = MagicMock()
    db_mongo.__getitem__.return_value = user_db
    db_mongo.list_database_names.return_value = ["appdb"]
    init_db(db_mongo)

    firecrawl = MagicMock()
    firecrawl.search.return_value = [
        {
            "url": "https://example.com/a",
            "title": "Privacy Policy",
            "description": "Data retention policy.",
        }
    ]
    init_core(MagicMock(), firecrawl, MagicMock())

    util_app = Flask("util-api-key")
    util_app.register_blueprint(util_bp)
    db_app = Flask("db-api-key")
    db_app.register_blueprint(db_bp)
    core_app = Flask("core-api-key")
    core_app.register_blueprint(core_bp)

    return {
        "util": util_app.test_client(),
        "db": db_app.test_client(),
        "core": core_app.test_client(),
        "util_coll": util_coll,
        "db_mongo": db_mongo,
        "user_coll": user_coll,
        "firecrawl": firecrawl,
    }


def _assert_handlers_not_called(gated_clients: dict[str, Any]) -> None:
    gated_clients["util_coll"].find.assert_not_called()
    gated_clients["db_mongo"].list_database_names.assert_not_called()
    gated_clients["user_coll"].delete_many.assert_not_called()
    gated_clients["firecrawl"].search.assert_not_called()


def test_init_app_api_key_strips_and_clears(monkeypatch: pytest.MonkeyPatch, restore_app_api_key) -> None:
    """Blank env disables the gate. A surrounding-whitespace value is stored stripped, including "0"."""
    security._app_api_key = "previous"
    monkeypatch.delenv("APP_API_KEY", raising=False)
    init_app_api_key()
    assert security._app_api_key is None

    for blank in ("", "   ", "\t\n"):
        security._app_api_key = "previous"
        monkeypatch.setenv("APP_API_KEY", blank)
        init_app_api_key()
        assert security._app_api_key is None

    monkeypatch.setenv("APP_API_KEY", "  secret  ")
    init_app_api_key()
    assert security._app_api_key == "secret"

    monkeypatch.setenv("APP_API_KEY", "0")
    init_app_api_key()
    assert security._app_api_key == "0"


def test_init_app_api_key_does_not_log_the_secret(
    monkeypatch: pytest.MonkeyPatch, restore_app_api_key, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("APP_API_KEY", "super-secret-value")
    with caplog.at_level(logging.DEBUG, logger="web-gather-api"):
        init_app_api_key()
    assert security._app_api_key == "super-secret-value"
    assert "super-secret-value" not in caplog.text


@pytest.mark.parametrize(
    ("client_name", "method", "path", "body"),
    [
        ("util", "get", "/embedding-models", None),
        ("db", "get", "/all-databases", None),
        ("db", "delete", "/documents?database_name=appdb&collection_name=docs", None),
        ("core", "post", "/gather", {"query": "privacy policy"}),
    ],
)
def test_configured_key_rejects_missing_header_before_handler(
    gated: dict[str, Any], client_name: str, method: str, path: str, body: dict | None
) -> None:
    security._app_api_key = "secret"
    kwargs = {"json": body} if body is not None else {}
    response = getattr(gated[client_name], method)(path, **kwargs)
    assert response.status_code == 401
    assert response.get_json() == {"error": "Missing API key."}
    _assert_handlers_not_called(gated)


def test_configured_key_rejects_blank_and_non_api_key_headers(gated: dict[str, Any]) -> None:
    """A whitespace key is missing, not invalid. Authorization is not a substitute for x-api-key."""
    security._app_api_key = "secret"
    blank = gated["util"].get("/embedding-models", headers={"x-api-key": "   "})
    assert blank.status_code == 401
    assert blank.get_json() == {"error": "Missing API key."}

    bearer = gated["db"].get("/all-databases", headers={"Authorization": "Bearer secret"})
    assert bearer.status_code == 401
    assert bearer.get_json() == {"error": "Missing API key."}
    _assert_handlers_not_called(gated)


@pytest.mark.parametrize("presented", ["wrong", "secre", "secret-extra", "Secret", "0"])
def test_configured_key_rejects_mismatched_key_before_handler(gated: dict[str, Any], presented: str) -> None:
    security._app_api_key = "secret"
    response = gated["util"].get("/embedding-models", headers={"x-api-key": presented})
    assert response.status_code == 403
    assert response.get_json() == {"error": "Invalid API key."}
    gated["util_coll"].find.assert_not_called()

    deleted = gated["db"].delete(
        "/documents?database_name=appdb&collection_name=docs",
        headers={"x-api-key": presented},
    )
    assert deleted.status_code == 403
    gated["user_coll"].delete_many.assert_not_called()


def test_matching_key_reaches_handler_after_header_normalization(gated: dict[str, Any]) -> None:
    """Padding is stripped and the header name is case-insensitive. Each blueprint stays gated."""
    security._app_api_key = "secret"
    headers = {"X-API-KEY": "  secret  "}

    models = gated["util"].get("/embedding-models", headers=headers)
    assert models.status_code == 200
    assert models.get_json()["models"][0]["model_name"] == "m"
    gated["util_coll"].find.assert_called_once()

    databases = gated["db"].get("/all-databases", headers=headers)
    assert databases.status_code == 200
    assert databases.get_json() == {"databases": ["appdb"]}
    gated["db_mongo"].list_database_names.assert_called_once()

    deleted = gated["db"].delete(
        "/documents?database_name=appdb&collection_name=docs",
        headers=headers,
    )
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted_count"] == 2
    gated["user_coll"].delete_many.assert_called_once()

    gathered = gated["core"].post("/gather", json={"query": "privacy policy"}, headers=headers)
    assert gathered.status_code == 200
    assert gathered.get_json()["query"] == "privacy policy"
    gated["firecrawl"].search.assert_called_once()


def test_unset_key_ignores_any_presented_header(gated: dict[str, Any]) -> None:
    security._app_api_key = None
    response = gated["core"].post(
        "/gather",
        json={"query": "privacy policy"},
        headers={"x-api-key": "not-configured"},
    )
    assert response.status_code == 200
    gated["firecrawl"].search.assert_called_once()

    deleted = gated["db"].delete(
        "/documents?database_name=appdb&collection_name=docs",
        headers={"x-api-key": ""},
    )
    assert deleted.status_code == 200
    gated["user_coll"].delete_many.assert_called_once()


def test_zero_key_is_enforced(gated: dict[str, Any]) -> None:
    security._app_api_key = "0"
    missing = gated["db"].get("/all-databases")
    assert missing.status_code == 401
    gated["db_mongo"].list_database_names.assert_not_called()

    wrong = gated["db"].get("/all-databases", headers={"x-api-key": "00"})
    assert wrong.status_code == 403
    gated["db_mongo"].list_database_names.assert_not_called()

    ok = gated["db"].get("/all-databases", headers={"x-api-key": "0"})
    assert ok.status_code == 200
    gated["db_mongo"].list_database_names.assert_called_once()


def test_async_api_key_decorator_matches_sync_contract(restore_app_api_key) -> None:
    """The unused async twin must keep the same accept/reject rules as require_api_key."""
    app = Flask("async-api-key")
    calls = {"n": 0}

    @require_api_key_async
    async def ping():
        calls["n"] += 1
        return jsonify({"ok": True}), 200

    def invoke(headers: dict[str, str] | None = None):
        with app.test_request_context("/async-ping", headers=headers or {}):
            return asyncio.run(ping())

    security._app_api_key = "secret"
    missing_body, missing_status = invoke()
    assert missing_status == 401
    assert missing_body.get_json() == {"error": "Missing API key."}

    blank_body, blank_status = invoke({"x-api-key": " \t "})
    assert blank_status == 401
    assert blank_body.get_json() == {"error": "Missing API key."}

    bad_body, bad_status = invoke({"x-api-key": "secret-extra"})
    assert bad_status == 403
    assert bad_body.get_json() == {"error": "Invalid API key."}
    assert calls["n"] == 0

    ok_body, ok_status = invoke({"X-Api-Key": "  secret  "})
    assert ok_status == 200
    assert ok_body.get_json() == {"ok": True}
    assert calls["n"] == 1

    security._app_api_key = None
    open_body, open_status = invoke({"x-api-key": "ignored"})
    assert open_status == 200
    assert open_body.get_json() == {"ok": True}
    assert calls["n"] == 2
