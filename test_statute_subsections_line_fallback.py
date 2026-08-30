"""Regression tests for statute-subsection line-range fallback and empty skips.

Valid start_line/end_line extraction is covered elsewhere. These tests pin:
- non-integer line ranges fall back to the LLM text field (not a skip/error)
- a subsection with neither identifier nor text is skipped
- identifier-only rows are still inserted
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


def _llm_text_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock_response.content = [block]
    return mock_response


def _wire_source_and_dest(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source_coll = MagicMock(name="source_coll")
    dest_coll = MagicMock(name="dest_coll")
    db = MagicMock(name="db")

    def _get_coll(name: str) -> MagicMock:
        return dest_coll if name == "statute_subchunks" else source_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    dest_coll.delete_many.return_value = MagicMock(deleted_count=0)
    return {"source_coll": source_coll, "dest_coll": dest_coll}


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, MagicMock(), mock_anthropic)
    return {"mongo": mock_mongo, "anthropic": mock_anthropic}


@pytest.fixture
def client(mock_clients) -> Any:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _payload() -> dict:
    return {
        "database": "db",
        "source_collection": "statute_chunks",
        "destination_collection": "statute_subchunks",
        "column": "chunk_text",
        "subsection_column": "subchunk_text",
    }


def test_invalid_line_range_falls_back_to_text_and_skips_empty(
    client, mock_clients
) -> None:
    dbs = _wire_source_and_dest(mock_clients["mongo"])
    dbs["source_coll"].find.return_value = [
        {"_id": "src-1", "chunk_text": "Plain statute text without a hash header."},
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"subsections":['
        '{"identifier":"(a)","text":"Fallback text from LLM.","start_line":"x","end_line":"y"},'
        '{"identifier":"","text":"","start_line":1,"end_line":0},'
        '{"identifier":"(c)","text":""}'
        "]}"
    )

    response = client.post("/create-statute-subsections", json=_payload())

    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 2
    assert data["source_rows_processed"] == 1
    assert data["source_rows_skipped"] == 0
    assert data["llm_errors"] == 0

    inserted = [call.args[0] for call in dbs["dest_coll"].insert_one.call_args_list]
    assert len(inserted) == 2
    assert inserted[0]["subsection_identifier"] == "(a)"
    assert inserted[0]["subchunk_text"] == "Fallback text from LLM."
    assert inserted[1]["subsection_identifier"] == "(c)"
    assert inserted[1]["subchunk_text"] == ""
    assert inserted[1]["source_id"] == "src-1"
