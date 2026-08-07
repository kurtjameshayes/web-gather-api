"""Behavioral regression tests for /create-paragraph-sections write semantics."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import core_bp, init_core


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


def _collections(mock_clients):
    """Return distinct source/dest collection mocks wired through mongo __getitem__."""
    source_coll = MagicMock()
    dest_coll = MagicMock()
    db = MagicMock()

    def _get_collection(name: str):
        if name == "chunks":
            return source_coll
        if name == "subchunks":
            return dest_coll
        return MagicMock()

    db.__getitem__.side_effect = _get_collection
    mock_clients["mongo"].__getitem__.return_value = db
    return source_coll, dest_coll


def test_create_paragraph_sections_applies_source_query_and_record_shape(client, mock_clients) -> None:
    """Writes one dest row per paragraph with source_id, original text, and collapsed whitespace."""
    source_coll, dest_coll = _collections(mock_clients)
    source_coll.find.return_value = [
        {
            "_id": "src-1",
            "document_id": "doc-9",
            "jurisdiction": "CA",
            "chunk_text": "Line   one.\n\nLine\ntwo.",
        }
    ]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "privacy-db",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
            "source_query": {"document_id": "doc-9"},
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 2
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 0
    source_coll.find.assert_called_once_with({"document_id": "doc-9"})

    assert dest_coll.insert_one.call_count == 2
    first = dest_coll.insert_one.call_args_list[0].args[0]
    second = dest_coll.insert_one.call_args_list[1].args[0]

    assert first["source_id"] == "src-1"
    assert first["document_id"] == "doc-9"
    assert first["jurisdiction"] == "CA"
    assert first["chunk_text"] == "Line   one.\n\nLine\ntwo."
    assert first["subchunk_text"] == "Line one."
    assert "subchunk_id" in first and first["subchunk_id"]
    assert "_id" not in first

    assert second["subchunk_text"] == "Line two."
    assert second["source_id"] == "src-1"
    assert second["subchunk_id"] != first["subchunk_id"]


def test_create_paragraph_sections_skips_empty_rows(client, mock_clients) -> None:
    """Rows with blank split-column values are skipped and counted."""
    source_coll, dest_coll = _collections(mock_clients)
    source_coll.find.return_value = [
        {"_id": "empty", "chunk_text": "   \n\n  "},
        {"_id": "missing", "other": "no chunk"},
        {"_id": "ok", "chunk_text": "Only paragraph."},
    ]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "privacy-db",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 1
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 2
    dest_coll.insert_one.assert_called_once()
    assert dest_coll.insert_one.call_args.args[0]["source_id"] == "ok"


def test_create_paragraph_sections_source_read_failure_returns_500(client, mock_clients) -> None:
    """Mongo read failures become 500 responses and do not write destination rows."""
    source_coll, dest_coll = _collections(mock_clients)
    source_coll.find.side_effect = RuntimeError("db down")

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "privacy-db",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )

    assert response.status_code == 500
    assert "Failed to read source collection" in response.get_json()["error"]
    dest_coll.insert_one.assert_not_called()
