"""Regression tests for /parse-llm ObjectId _id fallback lookup."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core


def _tool_use_response(sections: list[dict]) -> MagicMock:
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
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_parse_llm_falls_back_to_objectid_id_lookup(client, mock_clients) -> None:
    """When document_id misses, a valid ObjectId string must look up by _id."""
    oid = ObjectId()
    mock_collection = MagicMock()
    mock_collection.find_one.side_effect = [
        None,
        {"_id": oid, "text": "ObjectId-backed statute text."},
    ]
    mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

    mock_stream = MagicMock()
    mock_stream.get_final_message.return_value = _tool_use_response(
        [
            {
                "section": "§ 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "Content",
                "start_line": 1,
            }
        ]
    )
    mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

    response = client.post(
        "/parse-llm",
        json={
            "database": "test_db",
            "collection": "statutes",
            "parse_prompt": "Parse sections",
            "document_id": str(oid),
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert len(data["parsed_doc"]) == 1
    assert mock_collection.find_one.call_count == 2
    assert mock_collection.find_one.call_args_list[0][0][0] == {"document_id": str(oid)}
    assert mock_collection.find_one.call_args_list[1][0][0] == {"_id": oid}

    user_message = mock_clients["anthropic"].messages.stream.call_args[1]["messages"][0]["content"]
    assert "ObjectId-backed statute text." in user_message
