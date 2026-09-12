"""POST /create-paragraph-sections does not wipe the destination collection.

Statute subsection/subtopic jobs delete_many the destination before writing.
create-chunks also does not wipe (covered elsewhere). This pins the paragraph
path so a later shared-wipe helper cannot silently destroy existing subchunks.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_source_dest(mock_mongo: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Return distinct source/destination collection mocks on one database."""
    source_coll = MagicMock()
    dest_coll = MagicMock()
    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        if name == "subchunks":
            return dest_coll
        return source_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    return source_coll, dest_coll


def test_create_paragraph_sections_does_not_delete_destination(client, mock_clients) -> None:
    """Destination is append-only: no delete_many, even when source_query is omitted."""
    source_coll, dest_coll = _wire_source_dest(mock_clients)
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": "First paragraph.\n\nSecond paragraph."},
    ]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["records_inserted"] == 2
    dest_coll.delete_many.assert_not_called()
    dest_coll.delete_one.assert_not_called()
    source_coll.find.assert_called_once_with({})


def test_create_paragraph_sections_skips_empty_and_keeps_source_column(
    client, mock_clients
) -> None:
    """Blank source rows are skipped; destination rows keep the original split column."""
    source_coll, dest_coll = _wire_source_dest(mock_clients)
    source_coll.find.return_value = [
        {"_id": "empty", "chunk_text": "   ", "jurisdiction": "CA"},
        {"_id": "none", "chunk_text": None, "jurisdiction": "CA"},
        {
            "_id": "keep",
            "chunk_text": "Keep this paragraph.",
            "jurisdiction": "CA",
        },
    ]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_rows_skipped"] == 2
    assert payload["source_rows_processed"] == 1
    assert payload["records_inserted"] == 1
    dest_coll.delete_many.assert_not_called()

    dest_coll.insert_one.assert_called_once()
    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["subchunk_text"] == "Keep this paragraph."
    assert inserted["chunk_text"] == "Keep this paragraph."
    assert inserted["jurisdiction"] == "CA"
    assert inserted["source_id"] == "keep"
    assert "subchunk_id" in inserted
    assert "_id" not in inserted
