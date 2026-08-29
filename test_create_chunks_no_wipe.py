"""Regression tests for POST /create-chunks wipe and query contracts.

Unlike /create-statute-subsections and /create-statute-subtopics, create-chunks
does not delete the destination first and does not honor source_query.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, core_bp, init_core


@pytest.fixture
def dest_coll() -> MagicMock:
    return MagicMock()


@pytest.fixture
def source_coll() -> MagicMock:
    return MagicMock()


@pytest.fixture
def client(source_coll: MagicMock, dest_coll: MagicMock):
    mock_mongo = MagicMock()
    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        return dest_coll if name == "chunks" else source_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    mock_mongo.list_database_names.return_value = ["db"]
    db.list_collection_names.return_value = ["docs", "chunks"]
    init_core(mock_mongo, MagicMock(), MagicMock())

    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _required_body(**extra):
    body = {
        "database": "db",
        "source_collection": "docs",
        "destination_collection": "chunks",
        "source_column": "text",
        "chunk_column": "chunk_text",
    }
    body.update(extra)
    return body


def test_create_chunks_does_not_wipe_destination(client, source_coll, dest_coll) -> None:
    """create-chunks must not delete_many the destination before inserting."""
    source_coll.find.return_value = [
        {"_id": "src-1", "text": "short text", "keep": True},
    ]
    dest_coll.insert_one.return_value.inserted_id = "new-1"

    response = client.post("/create-chunks", json=_required_body(chunk_size=50, overlap=10))

    assert response.status_code == 200
    dest_coll.delete_many.assert_not_called()
    dest_coll.insert_one.assert_called()
    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["chunk_text"]
    assert inserted["chunk_index"] == 0
    assert inserted["source_id"] == "src-1"
    assert "text" not in inserted
    assert "_id" not in inserted
    assert inserted["keep"] is True


def test_create_chunks_ignores_source_query_and_finds_all(client, source_coll, dest_coll) -> None:
    """source_query in the body is ignored; source read is always find({})."""
    source_coll.find.return_value = []

    response = client.post(
        "/create-chunks",
        json=_required_body(source_query={"document_id": "only-this"}),
    )

    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})
    dest_coll.delete_many.assert_not_called()
    payload = response.get_json()
    assert payload["records_inserted"] == 0
    assert payload["source_rows_processed"] == 0


def test_create_chunks_omitted_size_uses_defaults(client, source_coll, dest_coll) -> None:
    """Omitting chunk_size/overlap echoes the module defaults (1200/200)."""
    source_coll.find.return_value = [{"_id": "1", "text": "hello world"}]

    response = client.post("/create-chunks", json=_required_body())

    assert response.status_code == 200
    data = response.get_json()
    assert data["chunk_size"] == DEFAULT_CHUNK_SIZE == 1200
    assert data["overlap"] == DEFAULT_CHUNK_OVERLAP == 200
    assert data["records_inserted"] == 1
