"""Regression tests for policy subsection parsing endpoint."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

from core import core_bp, init_core


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
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


def test_parse_policy_subsections_returns_subsections_and_counts(client, mock_clients) -> None:
    mock_collection = MagicMock()
    mock_collection.find.return_value = [
        {"_id": "doc-1", "chunk_text": "Policy text A", "category": "consumer_rights"},
        {"_id": "doc-2", "chunk_text": "   "},
    ]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

    with patch(
        "core._parse_policy_section_to_subsections",
        return_value=(
            [
                {
                    "subsection_identifier": "1",
                    "subsection_text": "Users may request deletion.",
                    "heading": "Deletion",
                    "category": "consumer_rights",
                    "start_line": 1,
                    "end_line": 2,
                }
            ],
            None,
        ),
    ):
        response = client.post(
            "/parse-policy-subsections",
            json={"database": "privacy-compliance", "collection": "policy_chunks", "column": "chunk_text"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database"] == "privacy-compliance"
    assert payload["collection"] == "policy_chunks"
    assert payload["column"] == "chunk_text"
    assert payload["source_rows_processed"] == 1
    assert payload["source_rows_skipped"] == 1
    assert payload["llm_errors"] == 0
    assert len(payload["subsections"]) == 1
    assert payload["subsections"][0]["source_id"] == "doc-1"
    assert payload["subsections"][0]["subsection_identifier"] == "1"
    assert payload["subsections"][0]["subsection_text"] == "Users may request deletion."


def test_parse_policy_subsections_invalid_source_query_json_returns_400(client, mock_clients) -> None:
    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "privacy-compliance",
            "collection": "policy_chunks",
            "column": "chunk_text",
            "source_query": "{bad-json}",
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "Invalid JSON in source_query" in payload["error"]
