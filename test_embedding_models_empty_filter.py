"""Regression tests for GET/POST /embedding-models empty-filter and body contracts.

GET treats a missing or empty database_name as unfiltered (query {}), but a
whitespace-only value is a real filter and is not stripped. POST uses
`get_json(silent=True) or {}`, so an empty body becomes missing-field 400
rather than "must be a JSON object"; a JSON array still hits the object check.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from util import init_util, util_bp


@pytest.fixture
def coll() -> MagicMock:
    mock_mongo = MagicMock()
    wg_db = MagicMock()
    coll = MagicMock()
    wg_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)
    coll.find.return_value = []
    return coll


@pytest.fixture
def client(coll: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def test_get_embedding_models_empty_database_name_is_unfiltered(client, coll) -> None:
    """?database_name= (empty string) is falsy and currently lists all models."""
    response = client.get("/embedding-models?database_name=")

    assert response.status_code == 200
    coll.find.assert_called_once_with({}, {"_id": 0})


def test_get_embedding_models_whitespace_database_name_is_a_filter(client, coll) -> None:
    """Whitespace is truthy and is used as the database_name filter, not stripped."""
    response = client.get("/embedding-models?database_name=%20")

    assert response.status_code == 200
    coll.find.assert_called_once_with({"database_name": " "}, {"_id": 0})


def test_post_embedding_models_empty_body_is_missing_database_name(client, coll) -> None:
    """Empty JSON object is coerced before the object-shape check, so the error is database_name."""
    response = client.post("/embedding-models", json={})

    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]
    coll.update_one.assert_not_called()


def test_post_embedding_models_json_array_is_rejected_as_non_object(client, coll) -> None:
    """A JSON array is truthy, so the validator reports that the body must be an object."""
    response = client.post(
        "/embedding-models",
        data="[1, 2]",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "json object" in response.get_json()["error"].lower()
    coll.update_one.assert_not_called()
