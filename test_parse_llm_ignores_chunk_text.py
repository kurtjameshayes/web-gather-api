"""POST /parse-llm concatenates only the `text` field.

Indexing writes `chunk_text` / `section_text`. parse-llm currently ignores those
columns, so a collection of indexed chunks with no `text` is treated as empty.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core
from test_core import create_tool_use_response


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, MagicMock(), mock_anthropic)
    return {"mongo": mock_mongo, "anthropic": mock_anthropic}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_parse_llm_chunk_text_only_is_no_text_content(client, mock_clients) -> None:
    """Indexed rows with chunk_text/section_text and no text never reach the LLM."""
    coll = MagicMock()
    coll.find.return_value = [
        {"_id": "1", "chunk_text": "Indexed statute chunk that must be ignored."},
        {"_id": "2", "section_text": "Section text that must be ignored."},
        {"_id": "3", "text": "   "},
    ]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = coll

    response = client.post(
        "/parse-llm",
        json={
            "database": "privacy-compliance",
            "collection": "statute_chunks",
            "parse_prompt": "Identify sections",
        },
    )

    assert response.status_code == 400
    assert "no text" in response.get_json()["error"].lower()
    mock_clients["anthropic"].messages.stream.assert_not_called()


def test_parse_llm_mixed_rows_send_only_text_field(client, mock_clients) -> None:
    """chunk_text on a row that also has text is omitted from the LLM document."""
    coll = MagicMock()
    coll.find.return_value = [
        {
            "_id": "1",
            "text": "Keep this statute text.",
            "chunk_text": "DO-NOT-SEND-CHUNK",
        },
        {"_id": "2", "chunk_text": "ALSO-DO-NOT-SEND"},
        {"_id": "3", "text": "  Second kept paragraph.  "},
    ]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = coll

    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = create_tool_use_response([])
    mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

    response = client.post(
        "/parse-llm",
        json={
            "database": "privacy-compliance",
            "collection": "statutes",
            "parse_prompt": "Identify sections",
        },
    )

    assert response.status_code == 200
    user_message = mock_clients["anthropic"].messages.stream.call_args[1]["messages"][0]["content"]
    assert "Keep this statute text." in user_message
    assert "Second kept paragraph." in user_message
    assert "DO-NOT-SEND-CHUNK" not in user_message
    assert "ALSO-DO-NOT-SEND" not in user_message
