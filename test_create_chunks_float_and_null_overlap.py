"""POST /create-chunks float coercion and null-overlap contracts.

#231 pins null chunk_size and non-integer string overlap. These leftovers cover
explicit null overlap (same 400 path) and JSON floats, which int() truncates
silently so chunking runs with a different size than the caller sent.
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


@pytest.fixture
def source_coll() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dest_coll() -> MagicMock:
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


_REQUIRED = {
    "database": "db",
    "source_collection": "docs",
    "destination_collection": "chunks",
    "source_column": "text",
    "chunk_column": "chunk_text",
}


def test_null_overlap_is_400_before_source_read(client, source_coll) -> None:
    """Explicit JSON null overlap is not the omitted default; int(None) is 400."""
    response = client.post("/create-chunks", json={**_REQUIRED, "overlap": None})

    assert response.status_code == 400
    assert "chunk_size and overlap must be integers" in response.get_json()["error"]
    source_coll.find.assert_not_called()


def test_float_chunk_size_and_overlap_are_truncated(client, source_coll, dest_coll) -> None:
    """JSON floats pass int() via truncation (50.9 -> 50, 2.9 -> 2), not 400."""
    source_coll.find.return_value = [
        {"_id": "1", "text": "abcdefghijklmnopqrstuvwxyz0123456789"},
    ]
    dest_coll.insert_one.return_value.inserted_id = "new-1"

    response = client.post(
        "/create-chunks",
        json={**_REQUIRED, "chunk_size": 50.9, "overlap": 2.9},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunk_size"] == 50
    assert payload["overlap"] == 2
    assert payload["records_inserted"] >= 1
    source_coll.find.assert_called_once_with({})
    dest_coll.insert_one.assert_called()
