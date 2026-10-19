"""POST /create-paragraph-sections write and query-scope edges.

$where forwarding on this route is covered elsewhere and is not retested.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from conftest import _wire_mongo_core
from core import core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _post(client, **overrides):
    payload = {
        "database": "db",
        "source_collection": "chunks",
        "destination_collection": "subchunks",
        "column": "chunk_text",
        "subsection_column": "subchunk_text",
    }
    payload.update(overrides)
    return client.post("/create-paragraph-sections", json=payload)


def test_paragraph_sections_collapse_subsection_text_and_keep_source_text(client, mock_clients) -> None:
    """Subsection text is whitespace-collapsed. The original column value is stored unchanged."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    collection = dbs["source_db"].__getitem__.return_value
    collection.find.return_value = [
        {
            "_id": "row-1",
            "chunk_text": "  Hello   world  \n\n\n\nSecond   line  ",
            "jurisdiction": "CA",
        },
        {"_id": "blank", "chunk_text": "   \n\n  "},
        {"_id": "missing", "other": "x"},
        {"_id": "zero", "chunk_text": 0},
        {"_id": "flag", "chunk_text": False},
    ]

    with patch("core.uuid.uuid4", side_effect=["u1", "u2", "u3", "u4"]):
        response = _post(client)

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 4
    assert body["source_rows_processed"] == 3
    assert body["source_rows_skipped"] == 2
    inserted = [call.args[0] for call in collection.insert_one.call_args_list]
    assert inserted[0] == {
        "jurisdiction": "CA",
        "source_id": "row-1",
        "subchunk_text": "Hello world",
        "subchunk_id": "u1",
        "chunk_text": "  Hello   world  \n\n\n\nSecond   line  ",
    }
    assert inserted[1]["subchunk_text"] == "Second line"
    assert inserted[1]["subchunk_id"] == "u2"
    assert inserted[1]["chunk_text"] == "  Hello   world  \n\n\n\nSecond   line  "
    assert inserted[2]["subchunk_text"] == "0"
    assert inserted[2]["chunk_text"] == "0"
    assert inserted[2]["source_id"] == "zero"
    assert inserted[3]["subchunk_text"] == "False"
    assert inserted[3]["chunk_text"] == "False"
    assert "blank" not in {row.get("source_id") for row in inserted}
    assert "missing" not in {row.get("source_id") for row in inserted}


def test_paragraph_sections_insert_error_stops_that_row_only(client, mock_clients) -> None:
    """A failed insert drops the rest of that source row. Later rows are still written."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    collection = dbs["source_db"].__getitem__.return_value
    collection.find.return_value = [
        {"_id": "a", "chunk_text": "One.\n\nTwo.\n\nThree."},
        {"_id": "b", "chunk_text": "Next only."},
    ]

    def insert(record):
        if record["subchunk_text"] == "Two.":
            raise RuntimeError("dup")

    collection.insert_one.side_effect = insert

    response = _post(client)

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 2
    assert body["source_rows_processed"] == 2
    assert body["source_rows_skipped"] == 0
    texts = [call.args[0]["subchunk_text"] for call in collection.insert_one.call_args_list]
    assert texts == ["One.", "Two.", "Next only."]
    sources = [call.args[0]["source_id"] for call in collection.insert_one.call_args_list]
    assert sources == ["a", "a", "b"]


@pytest.mark.parametrize("source_query", [["not-a-filter"], "[]", "   ", ""])
def test_paragraph_sections_non_object_source_query_is_unscoped(client, mock_clients, source_query) -> None:
    """A list, a JSON list, or a blank string does not 400. find() runs unfiltered."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    collection = dbs["source_db"].__getitem__.return_value
    collection.find.return_value = []

    response = _post(client, source_query=source_query)

    assert response.status_code == 200
    assert response.get_json()["records_inserted"] == 0
    assert response.get_json()["source_rows_skipped"] == 0
    collection.find.assert_called_once_with({})
    collection.insert_one.assert_not_called()


def test_paragraph_sections_padded_column_misses_the_field(client, mock_clients) -> None:
    """The column name is not stripped, so a padded name skips every row."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    collection = dbs["source_db"].__getitem__.return_value
    collection.find.return_value = [{"_id": "1", "chunk_text": "Hello"}]

    response = _post(client, column="  chunk_text  ")

    assert response.status_code == 200
    body = response.get_json()
    assert body["column"] == "  chunk_text  "
    assert body["records_inserted"] == 0
    assert body["source_rows_processed"] == 0
    assert body["source_rows_skipped"] == 1
    collection.insert_one.assert_not_called()


@pytest.mark.parametrize("column", [0, False, ""])
def test_paragraph_sections_falsy_column_is_missing(client, mock_clients, column) -> None:
    response = _post(client, column=column)

    assert response.status_code == 400
    assert "column" in response.get_json()["error"]
    mock_clients["mongo"].__getitem__.assert_not_called()


def test_paragraph_sections_padded_database_name_is_used(client, mock_clients) -> None:
    """A whitespace-padded database name is not rejected and is the Mongo database key."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    dbs["source_db"].__getitem__.return_value.find.return_value = []

    response = _post(client, database="  db  ")

    assert response.status_code == 200
    assert response.get_json()["database"] == "  db  "
    names = [call.args[0] for call in mock_clients["mongo"].__getitem__.call_args_list]
    assert "  db  " in names
