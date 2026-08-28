"""LLM row-skip after destination wipe for statute subsection/subtopic jobs.

Both endpoints delete destination rows matching source_query *before* calling
the LLM. A per-row LLM exception must skip that source document, continue later
rows, and still return HTTP 200 — otherwise a transient LLM failure either 500s
after data is already gone, or stops remaining statute indexing.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


def _llm_text(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _app_with_collections(source_coll: MagicMock, dest_coll: MagicMock, anthropic: MagicMock):
    mock_mongo = MagicMock()
    mock_db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        if name == "source_coll":
            return source_coll
        return dest_coll

    mock_db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = mock_db
    mock_mongo.list_database_names.return_value = ["privacy-compliance"]
    mock_db.list_collection_names.return_value = ["source_coll", "dest_coll"]
    init_core(mock_mongo, MagicMock(), anthropic)

    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client(), mock_db


def test_create_statute_subtopics_llm_failure_skips_row_after_wipe() -> None:
    """First-row LLM error is counted and skipped; later rows still insert."""
    source_coll = MagicMock()
    dest_coll = MagicMock()
    source_coll.find.return_value = [
        {"_id": "a", "chunk_text": "Notice at collection is required."},
        {"_id": "b", "chunk_text": "Consumers may request deletion."},
    ]
    dest_coll.delete_many.return_value.deleted_count = 4
    dest_coll.insert_one.return_value.inserted_id = "inserted-1"

    anthropic = MagicMock()
    anthropic.messages.create.side_effect = [
        RuntimeError("anthropic unavailable"),
        _llm_text('{"sub_topics": [{"sub_topic": "Right to delete"}]}'),
    ]
    client, mock_db = _app_with_collections(source_coll, dest_coll, anthropic)

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "privacy-compliance",
            "source_collection": "source_coll",
            "destination_collection": "dest_coll",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": {"document_id": "stat-1"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["llm_errors"] == 1
    assert payload["source_rows_skipped"] == 1
    assert payload["records_inserted"] == 1
    dest_coll.delete_many.assert_called_once_with({"document_id": "stat-1"})
    assert dest_coll.insert_one.call_count == 1
    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["sub_topic"] == "Right to delete"
    assert inserted["source_id"] == "b"
    assert anthropic.messages.create.call_count == 2
    mock_db.__getitem__.assert_any_call("source_coll")
    mock_db.__getitem__.assert_any_call("dest_coll")


def test_create_statute_subsections_llm_failure_skips_row_after_wipe() -> None:
    """Subsection jobs also wipe first, then continue after a per-row LLM error."""
    source_coll = MagicMock()
    dest_coll = MagicMock()
    source_coll.find.return_value = [
        {"_id": "a", "chunk_text": "(a) First duty.\n(b) Second duty."},
        {"_id": "b", "chunk_text": "(a) Remaining duty."},
    ]
    dest_coll.delete_many.return_value.deleted_count = 2
    dest_coll.insert_one.return_value.inserted_id = "inserted-1"

    anthropic = MagicMock()
    anthropic.messages.create.side_effect = [
        RuntimeError("timeout"),
        _llm_text(
            '{"subsections":[{"identifier":"(a)","start_line":1,"end_line":1,"text":"(a) Remaining duty."}]}'
        ),
    ]
    client, _mock_db = _app_with_collections(source_coll, dest_coll, anthropic)

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "privacy-compliance",
            "source_collection": "source_coll",
            "destination_collection": "dest_coll",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": {"document_id": "stat-2"},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["llm_errors"] == 1
    assert payload["source_rows_skipped"] == 1
    assert payload["records_inserted"] == 1
    dest_coll.delete_many.assert_called_once_with({"document_id": "stat-2"})
    assert dest_coll.insert_one.call_count == 1
    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["subsection_identifier"] == "(a)"
    assert inserted["source_id"] == "b"
