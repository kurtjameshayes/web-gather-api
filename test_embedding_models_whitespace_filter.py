"""Regression tests for GET /embedding-models empty vs whitespace database_name."""
from __future__ import annotations

from unittest.mock import MagicMock

from flask import Flask

from util import init_util, util_bp


def _client(mock_coll):
    wg_db = MagicMock()
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def test_embedding_models_empty_database_name_is_unfiltered() -> None:
    """``?database_name=`` is falsy, so the query is ``{}`` rather than an empty-string filter."""
    mock_coll = MagicMock()
    mock_coll.find.return_value = []
    client = _client(mock_coll)

    response = client.get("/embedding-models?database_name=")

    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({}, {"_id": 0})


def test_embedding_models_whitespace_database_name_is_used_as_is() -> None:
    """Whitespace is truthy and is not stripped before the catalog lookup."""
    mock_coll = MagicMock()
    mock_coll.find.return_value = []
    client = _client(mock_coll)

    response = client.get("/embedding-models?database_name=%20%20%20")

    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({"database_name": "   "}, {"_id": 0})
