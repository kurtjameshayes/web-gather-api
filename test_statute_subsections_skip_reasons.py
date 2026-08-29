"""Regression tests for POST /create-statute-subsections skip vs llm_error accounting.

Empty source text skips without an LLM call. Missing JSON increments llm_errors.
An empty subsections array skips without incrementing llm_errors.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import core_bp, init_core


def _llm_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


@pytest.fixture
def source_coll() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dest_coll() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_anthropic() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(source_coll: MagicMock, dest_coll: MagicMock, mock_anthropic: MagicMock):
    mock_mongo = MagicMock()
    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        return dest_coll if name == "statute_subchunks" else source_coll

    db.__getitem__.side_effect = _get_coll
    db.list_collection_names.return_value = ["statute_chunks", "statute_subchunks"]
    mock_mongo.__getitem__.return_value = db
    mock_mongo.list_database_names.return_value = ["db"]
    dest_coll.delete_many.return_value.deleted_count = 0
    init_core(mock_mongo, MagicMock(), mock_anthropic)

    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _body() -> dict:
    return {
        "database": "db",
        "source_collection": "statute_chunks",
        "destination_collection": "statute_subchunks",
        "column": "chunk_text",
        "subsection_column": "subchunk_text",
    }


def test_empty_and_whitespace_text_skips_without_llm(client, source_coll, dest_coll, mock_anthropic) -> None:
    source_coll.find.return_value = [
        {"_id": "1", "chunk_text": ""},
        {"_id": "2", "chunk_text": "   \n\t"},
        {"_id": "3", "chunk_text": None},
    ]

    response = client.post("/create-statute-subsections", json=_body())

    assert response.status_code == 200
    data = response.get_json()
    assert data["source_rows_skipped"] == 3
    assert data["source_rows_processed"] == 0
    assert data["records_inserted"] == 0
    assert data["llm_errors"] == 0
    mock_anthropic.messages.create.assert_not_called()
    dest_coll.insert_one.assert_not_called()


def test_no_json_block_counts_as_llm_error(client, source_coll, dest_coll, mock_anthropic) -> None:
    source_coll.find.return_value = [{"_id": "1", "chunk_text": "(a) A section."}]
    mock_anthropic.messages.create.return_value = _llm_response("sorry, I cannot produce JSON")

    response = client.post("/create-statute-subsections", json=_body())

    assert response.status_code == 200
    data = response.get_json()
    assert data["llm_errors"] == 1
    assert data["source_rows_skipped"] == 1
    assert data["source_rows_processed"] == 0
    assert data["records_inserted"] == 0
    dest_coll.insert_one.assert_not_called()


def test_empty_subsections_list_skips_without_llm_error(client, source_coll, dest_coll, mock_anthropic) -> None:
    source_coll.find.return_value = [{"_id": "1", "chunk_text": "(a) A section."}]
    mock_anthropic.messages.create.return_value = _llm_response('{"subsections": []}')

    response = client.post("/create-statute-subsections", json=_body())

    assert response.status_code == 200
    data = response.get_json()
    assert data["llm_errors"] == 0
    assert data["source_rows_skipped"] == 1
    assert data["source_rows_processed"] == 0
    assert data["records_inserted"] == 0
    dest_coll.insert_one.assert_not_called()
