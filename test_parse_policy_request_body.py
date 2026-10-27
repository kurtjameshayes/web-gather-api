"""Request-body parsing for POST /parse-policy-subsections.

`_get_json_payload_or_error` accepts a JSON object from either a JSON
content type or a raw body, and it turns non-object JSON values into an
empty object only when Flask already parsed them. A raw non-object is
returned as-is and then fails when the route reads fields.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


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


def _collection(mock_clients):
    coll = mock_clients["mongo"]["db"]["policy_chunks"]
    coll.find.return_value = []
    return coll


def test_json_content_type_non_object_is_missing_params(client, mock_clients) -> None:
    """A parsed JSON array, number, or false becomes {} and does not query."""
    coll = _collection(mock_clients)
    for body in ("[1, 2]", "5", "false"):
        coll.find.reset_mock()
        response = client.post(
            "/parse-policy-subsections",
            data=body,
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == (
            "Missing required parameters: database, collection, column"
        )
        coll.find.assert_not_called()


def test_raw_json_object_is_queried(client, mock_clients) -> None:
    """A JSON object without a JSON content type is still parsed and queried."""
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        data='{"database":"db","collection":"policy_chunks","column":"text"}',
        content_type="text/plain",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["database"] == "db"
    assert body["collection"] == "policy_chunks"
    assert body["column"] == "text"
    assert body["subsections"] == []
    coll.find.assert_called_once_with({})


def test_raw_json_string_is_not_coerced(client, mock_clients) -> None:
    """A raw JSON string is not turned into {} and the route returns 500."""
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        data='"hello"',
        content_type="text/plain",
    )
    assert response.status_code == 500
    coll.find.assert_not_called()


def test_invalid_json_is_400_before_query(client, mock_clients) -> None:
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        data="{",
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Invalid JSON in request body:")
    coll.find.assert_not_called()


def test_whitespace_body_is_missing_params(client, mock_clients) -> None:
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        data="   \n",
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == (
        "Missing required parameters: database, collection, column"
    )
    coll.find.assert_not_called()


def test_source_query_string_object_is_parsed(client, mock_clients) -> None:
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "db",
            "collection": "policy_chunks",
            "column": "text",
            "source_query": '{"document_id": "doc-9"}',
        },
    )
    assert response.status_code == 200
    coll.find.assert_called_once_with({"document_id": "doc-9"})


def test_source_query_non_object_becomes_unfiltered(client, mock_clients) -> None:
    """A list, a JSON list string, and whitespace scan the whole collection."""
    coll = _collection(mock_clients)
    payloads = (
        ["x"],
        "[]",
        "   ",
    )
    for source_query in payloads:
        coll.find.reset_mock()
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "db",
                "collection": "policy_chunks",
                "column": "text",
                "source_query": source_query,
            },
        )
        assert response.status_code == 200
        coll.find.assert_called_once_with({})


def test_source_query_invalid_string_is_400(client, mock_clients) -> None:
    coll = _collection(mock_clients)
    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "db",
            "collection": "policy_chunks",
            "column": "text",
            "source_query": "{",
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Invalid JSON in source_query:")
    coll.find.assert_not_called()
