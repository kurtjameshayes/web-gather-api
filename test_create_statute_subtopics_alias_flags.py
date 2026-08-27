"""POST /create-statute-subtopics LLM alias, flag coercion, and insert-failure contracts."""
from __future__ import annotations

import json
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
        if name == "statute_sections":
            return source_coll
        return dest_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    mock_mongo.list_database_names.return_value = ["privacy-compliance"]
    db.list_collection_names.return_value = ["statute_sections", "statute_subtopics"]
    return source_coll, dest_coll


def _llm_text_response(payload: dict) -> MagicMock:
    mock_response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    mock_response.content = [block]
    return mock_response


def test_subtopics_alias_and_flag_coercion(client, mock_clients) -> None:
    """Accepts LLM key 'subtopics', coerces flags with bool(), and string policy_categories become []."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {
            "_id": "sec-1",
            "chunk_text": "Consumers have the right to delete personal information.",
            "category": "consumer_rights",
            "jurisdiction": "CA",
        },
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        {
            "subtopics": [
                {
                    "sub_topic": "Right to delete",
                    "requirement_summary": "Must honor deletion requests.",
                    "policy_categories": "not-a-list",
                    "requires_consent": "false",
                    "consumer_facing": 0,
                },
                {
                    "sub_topic": "Notice of deletion",
                    "policy_categories": ["deletion", " notice "],
                    "requires_consent": True,
                    "consumer_facing": 1,
                },
            ]
        }
    )

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_sections",
            "destination_collection": "statute_subtopics",
            "column": "chunk_text",
            "subsection_column": "subsection_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 2
    assert data["source_rows_processed"] == 1
    assert dest_coll.insert_one.call_count == 2

    first = dest_coll.insert_one.call_args_list[0].args[0]
    assert first["sub_topic"] == "Right to delete"
    assert first["source_id"] == "sec-1"
    assert first["jurisdiction"] == "CA"
    assert first["category"] == "consumer_rights"
    assert first["requirement_summary"] == "Must honor deletion requests."
    assert first["policy_categories"] == []
    # bool("false") is True; bool(0) is False — pin current production coercion.
    assert first["requires_consent"] is True
    assert first["consumer_facing"] is False
    assert "chunk_text" not in first
    assert "_id" not in first
    assert "subchunk_id" in first
    assert first["subsection_text"]

    second = dest_coll.insert_one.call_args_list[1].args[0]
    assert second["sub_topic"] == "Notice of deletion"
    assert second["policy_categories"] == ["deletion", "notice"]
    assert second["requires_consent"] is True
    assert second["consumer_facing"] is True


def test_insert_failure_stops_remaining_subtopics_for_row(client, mock_clients) -> None:
    """insert_one failure breaks the inner subtopic loop and still returns 200."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {"_id": "sec-1", "chunk_text": "A testable consumer right to know."},
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        {
            "sub_topics": [
                {"sub_topic": "one"},
                {"sub_topic": "two"},
                {"sub_topic": "three"},
            ]
        }
    )
    dest_coll.insert_one.side_effect = [
        MagicMock(),
        RuntimeError("write error"),
        MagicMock(),
    ]

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_sections",
            "destination_collection": "statute_subtopics",
            "column": "chunk_text",
            "subsection_column": "subsection_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1
    assert dest_coll.insert_one.call_count == 2
    assert dest_coll.insert_one.call_args_list[0].args[0]["sub_topic"] == "one"
