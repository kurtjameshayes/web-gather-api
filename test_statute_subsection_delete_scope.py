"""Regression tests for statute subsection and subtopic destination wipes.

Both endpoints delete destination rows for `source_query` before any LLM call.
An empty or non-object query becomes `{}`, so delete_many removes every
destination row. A real filter must scope both the read and the delete.
Invalid JSON must fail before either call. A source read error must not delete.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from conftest import _wire_mongo_core
from core import core_bp, init_core

_ROUTES = ["/create-statute-subsections", "/create-statute-subtopics"]


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _collection(mock_clients):
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.find.return_value = []
    coll.delete_many.return_value.deleted_count = 0
    return coll


def _body(source_query: Any | None) -> dict:
    body = {
        "database": "statutes",
        "source_collection": "statute_chunks",
        "destination_collection": "statute_sub_chunks",
        "column": "chunk_text",
        "subsection_column": "sub_chunk_text",
    }
    if source_query is not None:
        body["source_query"] = source_query
    return body


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "source_query",
    [
        None,
        {},
        "   ",
        "[1, 2]",
        [1, 2],
        False,
        0,
    ],
)
def test_unscoped_query_wipes_destination_before_llm(
    client, mock_clients, route: str, source_query: Any
) -> None:
    """Missing or non-object queries read and delete with {} and skip the LLM."""
    coll = _collection(mock_clients)
    response = client.post(route, json=_body(source_query))
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["records_inserted"] == 0
    assert payload["source_rows_processed"] == 0
    assert payload["llm_errors"] == 0
    coll.find.assert_called_once_with({})
    coll.delete_many.assert_called_once_with({})
    mock_clients["anthropic"].messages.create.assert_not_called()


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "source_query",
    [
        {"document_id": "doc-9"},
        '{"document_id": "doc-9"}',
        {"document_id": ""},
        {"active": False},
    ],
)
def test_filter_scopes_read_and_delete(
    client, mock_clients, route: str, source_query: Any
) -> None:
    """A real filter is applied to find and delete_many, including JSON strings."""
    coll = _collection(mock_clients)
    expected = (
        {"document_id": "doc-9"}
        if source_query == '{"document_id": "doc-9"}'
        else source_query
    )
    response = client.post(route, json=_body(source_query))
    assert response.status_code == 200
    coll.find.assert_called_once_with(expected)
    coll.delete_many.assert_called_once_with(expected)
    mock_clients["anthropic"].messages.create.assert_not_called()


@pytest.mark.parametrize("route", _ROUTES)
def test_invalid_source_query_does_not_delete(client, mock_clients, route: str) -> None:
    """Invalid JSON is 400 before find or delete_many."""
    coll = _collection(mock_clients)
    response = client.post(route, json=_body("{bad"))
    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Invalid JSON in source_query:")
    coll.find.assert_not_called()
    coll.delete_many.assert_not_called()
    mock_clients["anthropic"].messages.create.assert_not_called()


@pytest.mark.parametrize("route", _ROUTES)
def test_source_read_error_does_not_delete(client, mock_clients, route: str) -> None:
    """A find failure returns 500 and leaves the destination collection alone."""
    coll = _collection(mock_clients)
    coll.find.side_effect = RuntimeError("mongo down")
    response = client.post(
        route,
        json=_body({"document_id": "doc-9"}),
    )
    assert response.status_code == 500
    assert response.get_json()["error"].startswith("Failed to read source collection:")
    coll.delete_many.assert_not_called()
    mock_clients["anthropic"].messages.create.assert_not_called()
