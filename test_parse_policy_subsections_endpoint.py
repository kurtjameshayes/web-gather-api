"""POST /parse-policy-subsections route behavior.

The LLM helper is covered elsewhere. These tests pin source_query application,
empty-row skips, LLM error continuation, source_id attachment, and the
no-write contract (results are returned, never inserted).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core
from conftest import _wire_mongo_core


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


def _source_coll(mock_clients) -> MagicMock:
    dbs = _wire_mongo_core(mock_clients["mongo"])
    return dbs["source_db"].__getitem__.return_value


def test_parse_policy_subsections_applies_source_query_and_does_not_write(client, mock_clients) -> None:
    coll = _source_coll(mock_clients)
    coll.find.return_value = [
        {"_id": "doc-1", "text": "We collect personal data."},
    ]
    parsed = [
        {
            "subsection_identifier": "1",
            "subsection_text": "We collect personal data.",
            "heading": "Collection",
            "category": "data_collection",
            "start_line": 1,
            "end_line": 1,
        }
    ]

    with patch("core._parse_policy_section_to_subsections", return_value=(parsed, None)) as mock_parse:
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "privacy-compliance",
                "collection": "policy_chunks",
                "column": "text",
                "source_query": {"document_id": "pol-1"},
                "parse_prompt": "Keep headings",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database"] == "privacy-compliance"
    assert payload["collection"] == "policy_chunks"
    assert payload["column"] == "text"
    assert payload["source_rows_processed"] == 1
    assert payload["source_rows_skipped"] == 0
    assert payload["llm_errors"] == 0
    assert payload["subsections"] == [
        {
            "subsection_identifier": "1",
            "subsection_text": "We collect personal data.",
            "heading": "Collection",
            "category": "data_collection",
            "start_line": 1,
            "end_line": 1,
            "source_id": "doc-1",
        }
    ]
    coll.find.assert_called_once_with({"document_id": "pol-1"})
    coll.insert_one.assert_not_called()
    coll.delete_many.assert_not_called()
    mock_parse.assert_called_once()
    args, kwargs = mock_parse.call_args
    assert args[0] == "We collect personal data."
    assert args[1] == "Keep headings"


def test_parse_policy_subsections_skips_empty_text_and_continues_after_llm_error(
    client, mock_clients
) -> None:
    coll = _source_coll(mock_clients)
    coll.find.return_value = [
        {"_id": "empty", "text": "   "},
        {"_id": "bad", "text": "Unparseable policy."},
        {"_id": "ok", "text": "Consumers may delete data."},
    ]
    parsed = [
        {
            "subsection_identifier": "rights",
            "subsection_text": "Consumers may delete data.",
            "heading": None,
            "category": "consumer_rights",
            "start_line": 1,
            "end_line": 1,
        }
    ]

    def _parse(text, parse_prompt="", doc=None):
        if "Unparseable" in text:
            return [], "No JSON block in LLM response"
        return parsed, None

    with patch("core._parse_policy_section_to_subsections", side_effect=_parse):
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "privacy-compliance",
                "collection": "policy_chunks",
                "column": "text",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_rows_processed"] == 1
    assert payload["source_rows_skipped"] == 2
    assert payload["llm_errors"] == 1
    assert len(payload["subsections"]) == 1
    assert payload["subsections"][0]["source_id"] == "ok"
    coll.insert_one.assert_not_called()


def test_parse_policy_subsections_invalid_source_query_json(client, mock_clients) -> None:
    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "privacy-compliance",
            "collection": "policy_chunks",
            "column": "text",
            "source_query": "{not-json}",
        },
    )

    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]
    mock_clients["mongo"].__getitem__.assert_not_called()


def test_parse_policy_subsections_non_object_source_query_defaults_to_empty(
    client, mock_clients
) -> None:
    """Non-object source_query is coerced to {} rather than 400 (current contract)."""
    coll = _source_coll(mock_clients)
    coll.find.return_value = []

    with patch("core._parse_policy_section_to_subsections", return_value=([], None)):
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "privacy-compliance",
                "collection": "policy_chunks",
                "column": "text",
                "source_query": ["not", "an", "object"],
            },
        )

    assert response.status_code == 200
    coll.find.assert_called_once_with({})


def test_parse_policy_subsections_source_read_failure_is_500(client, mock_clients) -> None:
    coll = _source_coll(mock_clients)
    coll.find.side_effect = RuntimeError("mongo down")

    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "privacy-compliance",
            "collection": "policy_chunks",
            "column": "text",
        },
    )

    assert response.status_code == 500
    assert "Failed to read source collection" in response.get_json()["error"]
