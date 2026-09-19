"""GET /search treats empty document_id/query as missing (400) but forwards
whitespace-only values to Mongo. Distinct from gather (#214) which also
accepts whitespace queries, and from alerts jurisdiction empty-vs-whitespace.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import WEB_GATHER_DB, core_bp, init_core
from conftest import _wire_mongo_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_search_empty_query_is_400(client, mock_clients) -> None:
    """query= is falsy, so the handler returns 400 before any Mongo lookup."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    docs = dbs["wg_db"].__getitem__.return_value

    response = client.get("/search?document_id=doc-1&query=")
    assert response.status_code == 400
    assert "document_id and query are required" in response.get_json()["error"]
    docs.find_one.assert_not_called()


def test_search_empty_document_id_is_400(client, mock_clients) -> None:
    """document_id= is falsy and is rejected the same way as a missing id."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    docs = dbs["wg_db"].__getitem__.return_value

    response = client.get("/search?document_id=&query=privacy")
    assert response.status_code == 400
    assert "document_id and query are required" in response.get_json()["error"]
    docs.find_one.assert_not_called()


def test_search_whitespace_document_id_is_forwarded(client, mock_clients) -> None:
    """Whitespace document_id is truthy and queried as-is (not stripped to empty)."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    docs = dbs["wg_db"].__getitem__.return_value
    docs.find_one.return_value = None

    response = client.get("/search?document_id=%20%20&query=privacy")
    assert response.status_code == 404
    assert "not found" in response.get_json()["error"]
    docs.find_one.assert_called_once_with({"document_id": "  "})


def test_search_whitespace_query_reaches_lookup(client, mock_clients) -> None:
    """Whitespace query is truthy, so document lookup still runs."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    docs = dbs["wg_db"].__getitem__.return_value
    docs.find_one.return_value = None

    response = client.get("/search?document_id=doc-1&query=%20%20")
    assert response.status_code == 404
    docs.find_one.assert_called_once_with({"document_id": "doc-1"})
