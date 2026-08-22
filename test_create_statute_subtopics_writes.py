"""Regression tests for /create-statute-subtopics write and parse contracts."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

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


def _llm_text_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock_response.content = [block]
    return mock_response


def _wire_source_dest(mock_mongo: MagicMock) -> tuple[MagicMock, MagicMock]:
    source_coll = MagicMock()
    dest_coll = MagicMock()
    dest_coll.delete_many.return_value = MagicMock(deleted_count=0)
    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        if name == "statute_sub_chunks":
            return source_coll
        return dest_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    return source_coll, dest_coll


def _valid_payload(**overrides):
    payload = {
        "database": "privacy-compliance",
        "source_collection": "statute_sub_chunks",
        "destination_collection": "statute_subtopics",
        "column": "chunk_text",
        "subsection_column": "sub_chunk_text",
    }
    payload.update(overrides)
    return payload


def test_create_statute_subtopics_empty_query_wipes_destination(client, mock_clients) -> None:
    """Omitting source_query currently deletes the entire destination collection."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {"_id": "src-1", "chunk_text": "A consumer may request deletion."}
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"sub_topics":[{"sub_topic":"Right to delete"}]}'
    )

    response = client.post("/create-statute-subtopics", json=_valid_payload())
    assert response.status_code == 200
    dest_coll.delete_many.assert_called_once_with({})
    dest_coll.insert_one.assert_called_once()
    inserted = dest_coll.insert_one.call_args.args[0]
    assert inserted["source_id"] == "src-1"
    assert inserted["sub_topic"] == "Right to delete"
    assert "chunk_text" not in inserted
    assert inserted["sub_chunk_text"] == "A consumer may request deletion."
    assert inserted["subchunk_id"]


def test_create_statute_subtopics_non_object_source_query_coerced_to_empty(
    client, mock_clients
) -> None:
    """A list/non-object source_query is coerced to {} and still wipes destination."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = []

    response = client.post(
        "/create-statute-subtopics",
        json=_valid_payload(source_query=["not-an-object"]),
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})
    dest_coll.delete_many.assert_called_once_with({})
    dest_coll.insert_one.assert_not_called()


def test_create_statute_subtopics_invalid_source_query_json_is_400(
    client, mock_clients
) -> None:
    response = client.post(
        "/create-statute-subtopics",
        json=_valid_payload(source_query="{bad"),
    )
    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]
    mock_clients["mongo"].__getitem__.assert_not_called()


def test_create_statute_subtopics_source_read_failure_is_500_without_inserts(
    client, mock_clients
) -> None:
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.side_effect = RuntimeError("mongo down")

    response = client.post("/create-statute-subtopics", json=_valid_payload())
    assert response.status_code == 500
    assert "Failed to read source collection" in response.get_json()["error"]
    dest_coll.insert_one.assert_not_called()


def test_create_statute_subtopics_parse_prompt_is_injected(client, mock_clients) -> None:
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {"_id": "src-2", "chunk_text": "Businesses must provide notice."}
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"sub_topics":[{"sub_topic":"Notice"}]}'
    )

    response = client.post(
        "/create-statute-subtopics",
        json=_valid_payload(parse_prompt="  Keep numeric markers together.  "),
    )
    assert response.status_code == 200
    prompt = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Additional parsing instructions:" in prompt
    assert "Keep numeric markers together." in prompt
    dest_coll.insert_one.assert_called_once()


def test_create_statute_subtopics_json5_fallback_and_empty_sub_topic_skip(
    client, mock_clients
) -> None:
    """Unparseable strict JSON falls back to json5; blank sub_topic entries are skipped."""
    source_coll, dest_coll = _wire_source_dest(mock_clients["mongo"])
    source_coll.find.return_value = [
        {"_id": "src-3", "chunk_text": "Consumers have a right to know."}
    ]
    # Trailing commas are invalid for json.loads but accepted by json5.
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        'Here you go: {"sub_topics":[{},{"sub_topic":""},{"sub_topic":"Right to know"},]}'
    )

    fake_json5 = MagicMock()
    fake_json5.loads.return_value = {
        "sub_topics": [
            {},
            {"sub_topic": ""},
            {"sub_topic": "Right to know"},
        ]
    }

    with patch.dict(sys.modules, {"json5": fake_json5}):
        response = client.post("/create-statute-subtopics", json=_valid_payload())

    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["llm_errors"] == 0
    fake_json5.loads.assert_called()
    dest_coll.insert_one.assert_called_once()
    assert dest_coll.insert_one.call_args.args[0]["sub_topic"] == "Right to know"
