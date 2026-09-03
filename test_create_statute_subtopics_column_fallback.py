"""Regression tests for POST /create-statute-subtopics empty-column fallback.

The endpoint silently substitutes another text field when the requested `column`
is missing or blank. That choice changes what the LLM sees and what is stored
as statute context, so it must stay pinned.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import core_bp, init_core


def _llm_json(payload: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [block]
    return response


def _wire_source_dest(mock_mongo: MagicMock, source_docs: list[dict]) -> tuple[MagicMock, MagicMock]:
    source_coll = MagicMock()
    dest_coll = MagicMock()
    source_coll.find.return_value = source_docs
    dest_coll.delete_many.return_value = MagicMock(deleted_count=0)

    db = MagicMock()
    collections = {"src": source_coll, "dest": dest_coll}
    db.__getitem__.side_effect = lambda name: collections[name]
    mock_mongo.__getitem__.return_value = db
    return source_coll, dest_coll


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


def _post_subtopics(client, **extra: Any):
    payload = {
        "database": "privacy-compliance",
        "source_collection": "src",
        "destination_collection": "dest",
        "column": "requested_col",
        "subsection_column": "sub_chunk_text",
    }
    payload.update(extra)
    return client.post("/create-statute-subtopics", json=payload)


def test_empty_requested_column_falls_back_to_sub_chunk_text(client, mock_clients) -> None:
    """Empty requested column uses sub_chunk_text before chunk_text."""
    _source_coll, dest_coll = _wire_source_dest(
        mock_clients["mongo"],
        [
            {
                "_id": "row-1",
                "requested_col": "   ",
                "sub_chunk_text": "FALLBACK_SUB_CHUNK",
                "chunk_text": "FALLBACK_CHUNK_TEXT",
                "category": "consumer_rights",
            }
        ],
    )
    mock_clients["anthropic"].messages.create.return_value = _llm_json(
        {
            "sub_topics": [
                {
                    "sub_topic": "right_to_delete",
                    "requirement_summary": "Consumers may request deletion.",
                }
            ]
        }
    )

    response = _post_subtopics(client)
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1
    assert data["source_rows_skipped"] == 0
    assert data["llm_errors"] == 0

    user_message = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "FALLBACK_SUB_CHUNK" in user_message
    assert "FALLBACK_CHUNK_TEXT" not in user_message
    assert "pre-classified into category: consumer_rights" in user_message

    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["sub_topic"] == "right_to_delete"
    assert inserted["sub_chunk_text"] == "FALLBACK_SUB_CHUNK"
    assert "requested_col" not in inserted
    assert inserted["source_id"] == "row-1"


def test_fallback_uses_chunk_text_when_sub_chunk_missing(client, mock_clients) -> None:
    """If sub_chunk_text is empty, fallback continues to chunk_text."""
    _wire_source_dest(
        mock_clients["mongo"],
        [
            {
                "_id": "row-2",
                "requested_col": None,
                "chunk_text": "FALLBACK_CHUNK_ONLY",
            }
        ],
    )
    mock_clients["anthropic"].messages.create.return_value = _llm_json(
        {"sub_topics": [{"sub_topic": "privacy_notice"}]}
    )

    response = _post_subtopics(client)
    assert response.status_code == 200
    assert response.get_json()["records_inserted"] == 1

    user_message = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "FALLBACK_CHUNK_ONLY" in user_message


def test_no_fallback_text_skips_without_llm(client, mock_clients) -> None:
    """Blank requested column and no fallback fields skip the row with no LLM call."""
    _wire_source_dest(
        mock_clients["mongo"],
        [{"_id": "row-3", "requested_col": "", "title": "no text fields"}],
    )

    response = _post_subtopics(client)
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 0
    assert data["source_rows_skipped"] == 1
    assert data["source_rows_processed"] == 0
    mock_clients["anthropic"].messages.create.assert_not_called()


def test_populated_requested_column_does_not_use_fallback(client, mock_clients) -> None:
    """A non-empty requested column is used even when fallback fields exist."""
    _wire_source_dest(
        mock_clients["mongo"],
        [
            {
                "_id": "row-4",
                "requested_col": "REQUESTED_COLUMN_TEXT",
                "sub_chunk_text": "SHOULD_NOT_APPEAR",
            }
        ],
    )
    mock_clients["anthropic"].messages.create.return_value = _llm_json(
        {"sub_topics": [{"sub_topic": "data_minimization"}]}
    )

    response = _post_subtopics(client)
    assert response.status_code == 200
    user_message = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "REQUESTED_COLUMN_TEXT" in user_message
    assert "SHOULD_NOT_APPEAR" not in user_message
