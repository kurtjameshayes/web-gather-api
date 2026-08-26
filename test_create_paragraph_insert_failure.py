"""Regression tests for POST /create-paragraph-sections write-failure and query parsing."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
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


def test_create_paragraph_sections_stops_row_on_insert_failure(client, mock_clients) -> None:
    """insert_one failure stops remaining subchunks for that source row; response is still 200."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = MagicMock()
    dest_coll = MagicMock()

    def _coll(name: str):
        return source_coll if name == "chunks" else dest_coll

    dbs["source_db"].__getitem__.side_effect = _coll
    source_coll.find.return_value = [
        {"_id": "src-1", "chunk_text": "Para one.\n\nPara two.\n\nPara three."},
    ]
    dest_coll.insert_one.side_effect = [MagicMock(), RuntimeError("duplicate key")]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "src",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1
    assert dest_coll.insert_one.call_count == 2


def test_create_paragraph_sections_parses_source_query_json_string(client, mock_clients) -> None:
    """source_query provided as a JSON object string is parsed and passed to find()."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = MagicMock()
    dest_coll = MagicMock()

    def _coll(name: str):
        return source_coll if name == "chunks" else dest_coll

    dbs["source_db"].__getitem__.side_effect = _coll
    source_coll.find.return_value = []

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "src",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
            "source_query": '{"document_id": "doc-1"}',
        },
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({"document_id": "doc-1"})
    dest_coll.insert_one.assert_not_called()


def test_create_paragraph_sections_coerces_non_object_source_query(client, mock_clients) -> None:
    """A JSON array source_query is coerced to {} rather than 400."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = MagicMock()
    dest_coll = MagicMock()

    def _coll(name: str):
        return source_coll if name == "chunks" else dest_coll

    dbs["source_db"].__getitem__.side_effect = _coll
    source_coll.find.return_value = []

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "src",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
            "source_query": [1, 2],
        },
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})
