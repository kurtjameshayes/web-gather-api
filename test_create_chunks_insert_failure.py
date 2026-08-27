"""POST /create-chunks insert failures stop remaining chunks for that source row only."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import core_bp, init_core


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


def _wire_source_dest(mock_mongo):
    source_coll = MagicMock()
    dest_coll = MagicMock()
    db = MagicMock()

    def _get_coll(name: str):
        if name == "docs":
            return source_coll
        return dest_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    mock_mongo.list_database_names.return_value = ["db"]
    db.list_collection_names.return_value = ["docs", "chunks"]
    return source_coll, dest_coll


def test_insert_failure_stops_remaining_chunks_for_that_row_only(client, mock_clients) -> None:
    """A failed insert_one breaks the inner chunk loop but still processes later source rows."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {"_id": "a", "text": "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC", "jurisdiction": "CA"},
        {"_id": "b", "text": "DDDDDDDDDDEEEEEEEEEE", "jurisdiction": "CA"},
    ]
    dest_coll.insert_one.side_effect = [
        MagicMock(),
        RuntimeError("duplicate key"),
        MagicMock(),
        MagicMock(),
    ]

    response = client.post(
        "/create-chunks",
        json={
            "database": "db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 10,
            "overlap": 0,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 3
    assert data["source_rows_processed"] == 2
    assert dest_coll.insert_one.call_count == 4

    first_payload = dest_coll.insert_one.call_args_list[0].args[0]
    assert first_payload["source_id"] == "a"
    assert first_payload["chunk_index"] == 0
    assert "text" not in first_payload
    assert "_id" not in first_payload
    assert first_payload["jurisdiction"] == "CA"
    assert first_payload["chunk_text"] == "AAAAAAAAAA"

    later_source_ids = [
        call.args[0]["source_id"] for call in dest_coll.insert_one.call_args_list[2:]
    ]
    assert later_source_ids == ["b", "b"]
