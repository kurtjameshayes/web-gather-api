"""Behavioral regression tests for /create-chunks write semantics."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

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
        if name == "docs":
            return source_coll
        if name == "chunks":
            return dest_coll
        return MagicMock()

    db.__getitem__.side_effect = _get_collection
    mock_clients["mongo"].__getitem__.return_value = db
    return source_coll, dest_coll


def test_create_chunks_record_shape_and_metadata(client, mock_clients) -> None:
    """Each chunk omits source text/_id and carries source_id, chunk_index, and other fields."""
    source_coll, dest_coll = _collections(mock_clients)
    # Long enough that chunk_size=20 with overlap yields multiple chunks.
    source_coll.find.return_value = [
        {
            "_id": "doc-42",
            "text": "abcdefghijklmnopqrstuvwxyz0123456789",
            "jurisdiction": "CA",
            "document_id": "policy-9",
        }
    ]

    response = client.post(
        "/create-chunks",
        json={
            "database": "privacy-db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 20,
            "overlap": 5,
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] >= 2
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 0
    source_coll.find.assert_called_once_with({})

    assert dest_coll.insert_one.call_count == body["records_inserted"]
    first = dest_coll.insert_one.call_args_list[0].args[0]
    second = dest_coll.insert_one.call_args_list[1].args[0]

    assert first["source_id"] == "doc-42"
    assert first["document_id"] == "policy-9"
    assert first["jurisdiction"] == "CA"
    assert first["chunk_index"] == 0
    assert "chunk_text" in first and first["chunk_text"]
    assert "text" not in first
    assert "_id" not in first

    assert second["chunk_index"] == 1
    assert second["source_id"] == "doc-42"
    assert second["chunk_text"] != first["chunk_text"]


def test_create_chunks_skips_empty_source_text(client, mock_clients) -> None:
    """Blank/missing source text is skipped and counted; valid rows still write."""
    source_coll, dest_coll = _collections(mock_clients)
    source_coll.find.return_value = [
        {"_id": "empty", "text": ""},
        {"_id": "missing", "other": "no text"},
        {"_id": "ok", "text": "Enough text to produce at least one chunk."},
    ]

    response = client.post(
        "/create-chunks",
        json={
            "database": "privacy-db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 50,
            "overlap": 0,
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["source_rows_skipped"] == 2
    assert body["source_rows_processed"] == 1
    assert body["records_inserted"] >= 1
    dest_coll.insert_one.assert_called()
    assert dest_coll.insert_one.call_args_list[0].args[0]["source_id"] == "ok"


def test_create_chunks_source_read_failure_returns_500(client, mock_clients) -> None:
    """Mongo read failures become 500 responses and do not write destination rows."""
    source_coll, dest_coll = _collections(mock_clients)
    source_coll.find.side_effect = RuntimeError("db down")

    response = client.post(
        "/create-chunks",
        json={
            "database": "privacy-db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 50,
            "overlap": 0,
        },
    )

    assert response.status_code == 500
    assert "Failed to read source collection" in response.get_json()["error"]
    dest_coll.insert_one.assert_not_called()
