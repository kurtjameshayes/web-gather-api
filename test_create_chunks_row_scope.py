"""POST /create-chunks row identity, falsy text, and partial insert failure."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from conftest import _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def client():
    mongo = MagicMock()
    init_core(mongo, MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client(), mongo


def _post(client, body):
    return client.post("/create-chunks", json=body)


def test_row_projection_falsy_text_and_defaults(client) -> None:
    """Copied fields drop _id and the source column; 0 and False are real text."""
    test_client, mongo = client
    dbs = _wire_mongo_core(mongo)
    source = dbs["source_db"].__getitem__.return_value
    source.find.return_value = [
        {"_id": "a", "text": None, "category": "drop-me"},
        {"_id": "b", "category": "missing-text"},
        {"_id": "c", "text": "   "},
        {"_id": "d", "text": 0, "category": "zero"},
        {"_id": "e", "text": False, "category": "bool"},
        {
            "_id": 15,
            "text": "Hello",
            "category": "rights",
            "source_id": "old",
            "chunk_index": 99,
        },
    ]

    response = _post(
        test_client,
        {
            "database": "  db  ",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["chunk_size"] == 1200
    assert payload["overlap"] == 200
    assert payload["records_inserted"] == 3
    assert payload["source_rows_processed"] == 3
    assert payload["source_rows_skipped"] == 3
    assert mongo.__getitem__.call_args_list[0].args[0] == "  db  "

    inserted = [call.args[0] for call in source.insert_one.call_args_list]
    by_source = {row["source_id"]: row for row in inserted}
    assert set(by_source) == {"d", "e", "15"}
    assert by_source["d"]["chunk_text"] == "0"
    assert by_source["d"]["category"] == "zero"
    assert by_source["d"]["chunk_index"] == 0
    assert "text" not in by_source["d"]
    assert "_id" not in by_source["d"]
    assert by_source["e"]["chunk_text"] == "False"
    hello = by_source["15"]
    assert hello["chunk_text"] == "Hello"
    assert hello["category"] == "rights"
    assert hello["chunk_index"] == 0
    assert hello["source_id"] == "15"
    assert "text" not in hello


def test_insert_error_stops_that_row_and_continues(client) -> None:
    """A failed insert keeps earlier chunks, skips the rest of the row, and continues."""
    test_client, mongo = client
    dbs = _wire_mongo_core(mongo)
    source = dbs["source_db"].__getitem__.return_value
    source.find.return_value = [
        {"_id": "row-1", "text": "ABCDEFGHIJ"},
        {"_id": "row-2", "text": "ZZZZZ"},
    ]
    calls = {"n": 0}

    def insert(doc):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("dup")

    source.insert_one.side_effect = insert

    response = _post(
        test_client,
        {
            "database": "db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 5,
            "overlap": 0,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["records_inserted"] == 2
    assert payload["source_rows_processed"] == 2
    assert payload["source_rows_skipped"] == 0
    attempted = [call.args[0] for call in source.insert_one.call_args_list]
    assert [row["chunk_text"] for row in attempted] == ["ABCDE", "FGHIJ", "ZZZZZ"]
    assert [row["source_id"] for row in attempted] == ["row-1", "row-1", "row-2"]
    assert attempted[0]["chunk_index"] == 0
    assert attempted[2]["chunk_index"] == 0


def test_source_read_error_is_500_before_insert(client) -> None:
    test_client, mongo = client
    dbs = _wire_mongo_core(mongo)
    source = dbs["source_db"].__getitem__.return_value
    source.find.side_effect = RuntimeError("disk")

    response = _post(
        test_client,
        {
            "database": "db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
        },
    )

    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to read source collection: disk"
    source.insert_one.assert_not_called()


def test_falsy_database_is_missing_before_mongo(client) -> None:
    test_client, mongo = client
    dbs = _wire_mongo_core(mongo)
    source = dbs["source_db"].__getitem__.return_value
    body = {
        "source_collection": "docs",
        "destination_collection": "chunks",
        "source_column": "text",
        "chunk_column": "chunk_text",
    }

    for database in (0, False):
        source.find.reset_mock()
        response = _post(test_client, {**body, "database": database})
        assert response.status_code == 400
        assert response.get_json()["error"] == "Missing required parameters: database"
        source.find.assert_not_called()
