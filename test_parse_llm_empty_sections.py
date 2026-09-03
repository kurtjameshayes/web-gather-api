"""Regression tests for POST /parse-llm empty tool-use sections.

Missing tool use is already a 500 on base. An identify_sections call that
returns `sections: []` is a successful empty parse, not an LLM error.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import core_bp, init_core


def _tool_use_response(sections) -> MagicMock:
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "identify_sections"
    tool_use_block.input = {"sections": sections}

    mock_response = MagicMock()
    mock_response.content = [tool_use_block]
    return mock_response


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_parse_llm_empty_sections_list_is_success(client, mock_clients) -> None:
    """Tool use with sections=[] returns HTTP 200 and an empty parsed_doc."""
    mock_collection = MagicMock()
    mock_collection.find.return_value = [{"_id": "1", "text": "Statute text that produced no sections."}]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = _tool_use_response([])
    mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

    response = client.post(
        "/parse-llm",
        json={
            "database": "test_db",
            "collection": "test_collection",
            "parse_prompt": "Parse sections",
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {"parsed_doc": []}


def test_parse_llm_missing_sections_key_is_success(client, mock_clients) -> None:
    """Omitted `sections` defaults to [] and is still a 200, not a 500."""
    mock_collection = MagicMock()
    mock_collection.find.return_value = [{"_id": "1", "text": "Statute text."}]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "identify_sections"
    tool_use_block.input = {}
    mock_response = MagicMock()
    mock_response.content = [tool_use_block]
    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = mock_response
    mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

    response = client.post(
        "/parse-llm",
        json={
            "database": "test_db",
            "collection": "test_collection",
            "parse_prompt": "Parse sections",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["parsed_doc"] == []
