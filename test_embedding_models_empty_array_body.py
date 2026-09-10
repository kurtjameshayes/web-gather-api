"""POST /embedding-models empty vs non-empty JSON array body contracts."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import init_db
from util import init_util, util_bp


@pytest.fixture
def client():
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def test_empty_json_array_becomes_missing_database_name(client) -> None:
    """`[]` is falsy, so `get_json() or {}` yields missing database_name (not object-type)."""
    response = client.post("/embedding-models", json=[])
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]
    assert "JSON object" not in response.get_json()["error"]


def test_nonempty_json_array_must_be_object(client) -> None:
    """A truthy JSON array is rejected as a non-object body before field checks."""
    response = client.post(
        "/embedding-models",
        json=[{"database_name": "privacy-compliance", "model_name": "x", "fields": []}],
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must be a JSON object"
