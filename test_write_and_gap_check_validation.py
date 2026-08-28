"""Validation-order contracts for write_to_collection and /gap-check.

These paths currently fail *before* the documented required-field checks:
JSON `null` bodies crash write_to_collection, and a non-integer
max_policy_chars crashes gap-check. Pin both so a later 400-guard is an
intentional contract change, not an accidental behavior drift.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core
from db import db_bp, init_db


def test_write_to_collection_json_null_is_generic_500() -> None:
    """POST /write_to_collection with JSON null currently raises (generic 500)."""
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    client = app.test_client()

    response = client.post(
        "/write_to_collection",
        data="null",
        content_type="application/json",
    )

    assert response.status_code == 500
    # Uncaught AttributeError is not mapped to the required-field 400 JSON body.
    payload = response.get_json(silent=True)
    if payload is not None:
        assert "database_name" not in str(payload.get("error", "")).lower()
    mock_mongo.__getitem__.assert_not_called()


def test_write_to_collection_empty_object_is_400() -> None:
    """POST /write_to_collection with {} still hits the required-field 400."""
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    client = app.test_client()

    response = client.post("/write_to_collection", json={})

    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()


def test_gap_check_invalid_max_policy_chars_is_generic_500() -> None:
    """Non-integer max_policy_chars is int()'d before required-param checks."""
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    client = app.test_client()

    response = client.post(
        "/gap-check",
        json={"max_policy_chars": "not-an-int"},
    )

    assert response.status_code == 500
    payload = response.get_json(silent=True)
    if payload is not None:
        err = str(payload.get("error", "")).lower()
        assert "required" not in err
    mock_mongo.__getitem__.assert_not_called()


def test_gap_check_missing_params_is_400_when_max_policy_chars_omitted() -> None:
    """Without max_policy_chars, missing policy/statute ids still 400 as documented."""
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    client = app.test_client()

    response = client.post("/gap-check", json={})

    assert response.status_code == 400
    assert "policy_database_name" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()
