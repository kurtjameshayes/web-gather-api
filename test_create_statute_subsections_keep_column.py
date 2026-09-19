"""POST /create-statute-subsections keeps the original split column on dest rows.

Subtopic jobs omit the requested split column (#213). Subsection jobs copy
the full source text back onto that column plus subsection_column /
subsection_identifier. A later 'align the two writers' refactor would drop
the source text from every statute subsection used by embeddings.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import WEB_GATHER_DB, core_bp, init_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, MagicMock(), mock_anthropic)
    return {"mongo": mock_mongo, "anthropic": mock_anthropic}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_source_and_dest(mock_mongo: MagicMock):
    source_coll = MagicMock()
    dest_coll = MagicMock()
    user_db = MagicMock()
    wg_db = MagicMock()

    def _get_db(name: str):
        return wg_db if name == WEB_GATHER_DB else user_db

    def _get_coll(name: str):
        if name == "statute_chunks":
            return source_coll
        if name == "statute_subchunks":
            return dest_coll
        return MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    user_db.__getitem__.side_effect = _get_coll
    wg_db.__getitem__.return_value = MagicMock()
    dest_coll.delete_many.return_value.deleted_count = 0
    return source_coll, dest_coll


def test_create_statute_subsections_keeps_split_column(client, mock_clients) -> None:
    """Inserted rows keep chunk_text (the split column) plus subchunk_text."""
    source_coll, dest_coll = _wire_source_and_dest(mock_clients["mongo"])
    source_text = "(a) Controllers shall disclose. (b) Consumers may delete."
    source_coll.find.return_value = [
        {"_id": "row-1", "chunk_text": source_text, "jurisdiction": "CA"},
    ]

    block = MagicMock()
    block.type = "text"
    block.text = (
        '{"subsections":[{"identifier":"(a)","text":"Controllers shall disclose."}]}'
    )
    mock_response = MagicMock()
    mock_response.content = [block]
    mock_clients["anthropic"].messages.create.return_value = mock_response

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    dest_coll.insert_one.assert_called_once()
    record = dest_coll.insert_one.call_args[0][0]
    assert record["chunk_text"] == source_text
    assert record["subchunk_text"] == "Controllers shall disclose."
    assert record["subsection_identifier"] == "(a)"
    assert record["jurisdiction"] == "CA"
    assert record["source_id"] == "row-1"
    assert "subchunk_id" in record
