"""Regression tests for /parse-policy-subsections request-body JSON contract.

This endpoint is the only caller of core._get_json_payload_or_error. Invalid
bodies must 400 before Mongo is touched; non-object or empty bodies are treated
as {} and then fail the required-field check.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_mongo() -> MagicMock:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


@pytest.fixture
def client(mock_mongo: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_parse_policy_subsections_invalid_body_json_is_400(client, mock_mongo) -> None:
    """Malformed request JSON is 400 and must not query Mongo."""
    response = client.post(
        "/parse-policy-subsections",
        data='{"database": "db",',
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Invalid JSON in request body" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()


def test_parse_policy_subsections_json_array_body_is_missing_params(
    client, mock_mongo
) -> None:
    """A JSON array is coerced to {} (not 400-invalid) and then fails required fields."""
    response = client.post(
        "/parse-policy-subsections",
        data="[]",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()


def test_parse_policy_subsections_empty_body_is_missing_params(
    client, mock_mongo
) -> None:
    """An empty body is treated as {} and returns the required-field 400."""
    response = client.post(
        "/parse-policy-subsections",
        data="",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]
    mock_mongo.__getitem__.assert_not_called()
