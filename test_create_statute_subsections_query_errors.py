"""Regression tests for leftover /create-statute-subsections query and parse edges.

PR #198 covers line-range extraction, parse_prompt injection, scoped delete_many,
and non-fatal LLM failures. These tests pin the remaining wipe/parse contracts:
empty/non-object source_query currently deletes the whole destination,
source-read failures must not write, trailing-comma LLM JSON falls back to
JSON5, and a destination insert error stops that source row without 500.
"""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core  # noqa: E402


def _llm_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _wire_source_and_dest(mock_mongo: MagicMock) -> Dict[str, MagicMock]:
    source = MagicMock(name="source")
    dest = MagicMock(name="dest")
    dest.delete_many.return_value = MagicMock(deleted_count=0)
    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        if name == "statute_sub_chunks":
            return dest
        return source

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    return {"source": source, "dest": dest}


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


def _required_payload(**overrides: Any) -> dict:
    payload = {
        "database": "privacy-compliance",
        "source_collection": "statute_chunks",
        "destination_collection": "statute_sub_chunks",
        "column": "chunk_text",
        "subsection_column": "sub_chunk_text",
    }
    payload.update(overrides)
    return payload


def test_create_statute_subsections_omitted_source_query_wipes_destination(
    client, mock_clients
) -> None:
    """Omitting source_query currently calls delete_many({}) on the destination."""
    colls = _wire_source_and_dest(mock_clients["mongo"])
    colls["source"].find.return_value = []

    response = client.post("/create-statute-subsections", json=_required_payload())

    assert response.status_code == 200
    colls["source"].find.assert_called_once_with({})
    colls["dest"].delete_many.assert_called_once_with({})
    colls["dest"].insert_one.assert_not_called()


def test_create_statute_subsections_non_object_source_query_coerced_to_empty(
    client, mock_clients
) -> None:
    """A list source_query is coerced to {} rather than rejected with 400."""
    colls = _wire_source_and_dest(mock_clients["mongo"])
    colls["source"].find.return_value = []

    response = client.post(
        "/create-statute-subsections",
        json=_required_payload(source_query=["document_id", "stat-1"]),
    )

    assert response.status_code == 200
    colls["source"].find.assert_called_once_with({})
    colls["dest"].delete_many.assert_called_once_with({})


def test_create_statute_subsections_source_read_failure_does_not_delete_or_insert(
    client, mock_clients
) -> None:
    """A source find() exception is HTTP 500 and must not wipe the destination."""
    colls = _wire_source_and_dest(mock_clients["mongo"])
    colls["source"].find.side_effect = RuntimeError("socket closed")

    response = client.post("/create-statute-subsections", json=_required_payload())

    assert response.status_code == 500
    assert "Failed to read source collection" in response.get_json()["error"]
    colls["dest"].delete_many.assert_not_called()
    colls["dest"].insert_one.assert_not_called()
    mock_clients["anthropic"].messages.create.assert_not_called()


def test_create_statute_subsections_json5_trailing_comma_fallback(
    client, mock_clients, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trailing-comma LLM JSON is accepted via json5 without patching core.json.loads."""
    import json as stdjson

    class _FakeJson5:
        @staticmethod
        def loads(raw: str):
            cleaned = raw.replace(",}", "}").replace(",]", "]")
            return stdjson.loads(cleaned)

    monkeypatch.setitem(sys.modules, "json5", _FakeJson5())

    colls = _wire_source_and_dest(mock_clients["mongo"])
    colls["source"].find.return_value = [
        {"_id": "src-1", "chunk_text": "(a) A consumer may request deletion."},
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"subsections":[{"identifier":"(a)","text":"A consumer may request deletion.",},]}'
    )

    response = client.post("/create-statute-subsections", json=_required_payload())

    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["llm_errors"] == 0
    inserted = colls["dest"].insert_one.call_args.args[0]
    assert inserted["subsection_identifier"] == "(a)"
    assert inserted["source_id"] == "src-1"
    assert "A consumer may request deletion." in inserted["sub_chunk_text"]


def test_create_statute_subsections_insert_failure_stops_row_without_500(
    client, mock_clients
) -> None:
    """A destination insert error breaks that source row but still returns 200."""
    colls = _wire_source_and_dest(mock_clients["mongo"])
    colls["source"].find.return_value = [
        {"_id": "src-1", "chunk_text": "(a) First. (b) Second."},
    ]
    colls["dest"].insert_one.side_effect = RuntimeError("duplicate key")
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"subsections":['
        '{"identifier":"(a)","text":"First.","start_line":1,"end_line":1},'
        '{"identifier":"(b)","text":"Second.","start_line":1,"end_line":1}'
        "]}"
    )

    response = client.post("/create-statute-subsections", json=_required_payload())

    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 0
    assert data["source_rows_processed"] == 1
    assert data["llm_errors"] == 0
    assert colls["dest"].insert_one.call_count == 1
