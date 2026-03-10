"""Regression tests for parse-policy-subsections endpoint and helper."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from core import _parse_policy_section_to_subsections, core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


def test_parse_policy_helper_extracts_line_ranges_and_doc_category(mock_clients) -> None:
    text = "Policy title\nWe collect usage data.\nWe retain it for one year."
    llm_json = (
        '{"sections": [{"identifier": "1", "heading": "Data retention", '
        '"start_line": 2, "end_line": 3}]}'
    )
    mock_clients["anthropic"].messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=llm_json)]
    )

    subsections, err = _parse_policy_section_to_subsections(
        text,
        parse_prompt="Focus on retention language.",
        doc={"category": "state_specific"},
    )

    assert err is None
    assert len(subsections) == 1
    assert subsections[0]["subsection_identifier"] == "1"
    assert subsections[0]["heading"] == "Data retention"
    assert subsections[0]["subsection_text"] == "We collect usage data. We retain it for one year."
    assert subsections[0]["category"] == "state_specific"

    call_args = mock_clients["anthropic"].messages.create.call_args.kwargs
    assert "Additional parsing instructions" in call_args["messages"][0]["content"]
    assert "Focus on retention language." in call_args["messages"][0]["content"]


def test_parse_policy_helper_returns_error_when_json_missing(mock_clients) -> None:
    mock_clients["anthropic"].messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="No structured output")]
    )

    subsections, err = _parse_policy_section_to_subsections("Some text")

    assert subsections == []
    assert err == "No JSON block in LLM response"


def test_parse_policy_subsections_endpoint_tracks_counts_and_source_ids(client, mock_clients) -> None:
    mock_clients["mongo"].list_database_names.return_value = ["test_db"]
    db = MagicMock()
    source_coll = MagicMock()
    source_coll.find.return_value = [
        {"_id": "doc-1", "chunk_text": "First policy section."},
        {"_id": "doc-2", "chunk_text": "Second policy section."},
        {"_id": "doc-3", "chunk_text": "   "},
    ]
    db.__getitem__.return_value = source_coll
    mock_clients["mongo"].__getitem__.return_value = db

    with patch(
        "core._parse_policy_section_to_subsections",
        side_effect=[
            (
                [
                    {
                        "subsection_identifier": "A",
                        "subsection_text": "First parsed subsection",
                        "heading": "Intro",
                        "category": "consumer_rights",
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
                None,
            ),
            ([], "No JSON block in LLM response"),
        ],
    ):
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "test_db",
                "collection": "policy_chunks",
                "column": "chunk_text",
                "source_query": '{"document_id": "policy-1"}',
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_rows_processed"] == 1
    assert payload["source_rows_skipped"] == 2
    assert payload["llm_errors"] == 1
    assert len(payload["subsections"]) == 1
    assert payload["subsections"][0]["source_id"] == "doc-1"
    source_coll.find.assert_called_once_with({"document_id": "policy-1"})


def test_parse_policy_subsections_rejects_invalid_source_query_json(client, mock_clients) -> None:
    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "test_db",
            "collection": "policy_chunks",
            "column": "chunk_text",
            "source_query": "{invalid-json",
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "Invalid JSON in source_query" in payload["error"]
