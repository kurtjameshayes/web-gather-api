"""Statute subsection line binding and subtopic record mapping.

These endpoints write the corpus later gap analysis reads. The route tests only
check missing parameters, so the mapping rules below are otherwise unpinned.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

from flask import Flask

from conftest import _wire_mongo_core
from core import core_bp, init_core


def _app_client():
    mongo = MagicMock()
    mongo.list_database_names.return_value = ["db"]
    anthropic = MagicMock()
    init_core(mongo, MagicMock(), anthropic)
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    dbs = _wire_mongo_core(mongo)
    coll = dbs["source_db"].__getitem__.return_value
    coll.list_collection_names.return_value = ["statute_chunks", "statute_sub_chunks"]
    return app.test_client(), coll, anthropic


def _text_blocks(*texts: str) -> MagicMock:
    response = MagicMock()
    blocks = []
    tool = MagicMock()
    tool.type = "tool_use"
    blocks.append(tool)
    for text in texts:
        block = MagicMock()
        block.type = "text"
        block.text = text
        blocks.append(block)
    response.content = blocks
    return response


def _messages(anthropic: MagicMock) -> list[str]:
    return [call.kwargs["messages"][0]["content"] for call in anthropic.messages.create.call_args_list]


class _Json5:
    def __init__(self) -> None:
        self.raw: list[str] = []

    def loads(self, raw: str):
        self.raw.append(raw)
        if "subsections" in raw:
            return {
                "subsections": [
                    {"identifier": "(a)", "start_line": 1, "end_line": 1, "text": "ignored"}
                ]
            }
        return {"sub_topics": [{"sub_topic": "retention_disclosure"}]}


def _install_json5() -> tuple[_Json5, object]:
    stub = _Json5()
    previous = sys.modules.get("json5")
    sys.modules["json5"] = stub
    return stub, previous


def _restore_json5(previous: object) -> None:
    if previous is None:
        sys.modules.pop("json5", None)
    else:
        sys.modules["json5"] = previous


def test_statute_subsections_bind_line_ranges_and_chunk_header() -> None:
    """Inclusive line ranges win over copied text, and the chunk header is prepended once."""
    client, coll, anthropic = _app_client()
    statute = "# 1798.100 Title\n\n(a) First   line\n\n(b) Second"
    inserted: list[dict] = []
    coll.find.return_value = [
        {"_id": 15, "chunk_text": statute, "jurisdiction": "CA", "extra": None},
        {"_id": "blank", "chunk_text": "   ", "jurisdiction": "CA"},
    ]
    coll.insert_one.side_effect = lambda doc: inserted.append(doc)
    anthropic.messages.create.return_value = _text_blocks(
        json.dumps(
            {
                "subsections": [
                    {
                        "identifier": "(a)",
                        "text": "should not win",
                        "start_line": 1,
                        "end_line": 3,
                        "category": " consumer_rights ",
                        "header_text": "  Right  ",
                        "category_reasoning": "  because  ",
                    },
                    {
                        "identifier": "(b)",
                        "start_line": 5,
                        "end_line": 5,
                    },
                    {"identifier": "", "text": ""},
                    {
                        "identifier": "(c)",
                        "text": "  kept   quote  ",
                        "start_line": 9,
                        "end_line": 1,
                    },
                    {
                        "identifier": "(d)",
                        "text": "fallback",
                        "start_line": "nope",
                        "end_line": 1,
                    },
                ]
            }
        )
    )

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "db",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_sub_chunks",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "parse_prompt": "   ",
            "source_query": {"document_id": "sec-1"},
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 4
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 1
    assert body["llm_errors"] == 0
    assert [row["subsection_identifier"] for row in inserted] == ["(a)", "(b)", "(c)", "(d)"]
    assert inserted[0]["sub_chunk_text"] == "# 1798.100 Title (a) First line"
    assert inserted[1]["sub_chunk_text"] == "# 1798.100 Title (b) Second"
    assert inserted[2]["sub_chunk_text"] == "# 1798.100 Title kept quote"
    assert inserted[3]["sub_chunk_text"] == "# 1798.100 Title fallback"
    assert inserted[0]["category"] == "consumer_rights"
    assert inserted[0]["header_text"] == "Right"
    assert inserted[0]["category_reasoning"] == "because"
    assert "header_text" not in inserted[1]
    assert inserted[0]["source_id"] == "15"
    assert inserted[0]["jurisdiction"] == "CA"
    assert inserted[0]["extra"] is None
    assert inserted[0]["chunk_text"] == statute
    assert "_id" not in inserted[0]
    assert all(len(row["subchunk_id"]) == 36 for row in inserted)
    coll.find.assert_called_with({"document_id": "sec-1"})
    coll.delete_many.assert_called_with({"document_id": "sec-1"})
    prompt = _messages(anthropic)[0]
    assert "Additional parsing instructions" not in prompt
    assert "1: # 1798.100 Title" in prompt
    assert anthropic.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-6"
    assert anthropic.messages.create.call_args.kwargs["max_tokens"] == 8192


def test_statute_subsections_llm_failure_continues_and_json5_parses() -> None:
    """A failed model call does not stop the next row, and trailing-comma JSON uses json5."""
    client, coll, anthropic = _app_client()
    inserted: list[dict] = []
    coll.find.return_value = [
        {"_id": "err", "chunk_text": "first"},
        {"_id": "noise", "chunk_text": "second"},
        {"_id": "ok", "chunk_text": "Alpha line"},
    ]
    coll.insert_one.side_effect = lambda doc: inserted.append(doc)
    anthropic.messages.create.side_effect = [
        RuntimeError("overloaded"),
        _text_blocks("no object here"),
        _text_blocks('preamble {"subsections":[{"identifier":"(a)","text":"copied",}]} tail'),
    ]
    stub, previous = _install_json5()
    try:
        response = client.post(
            "/create-statute-subsections",
            json={
                "database": "db",
                "source_collection": "statute_chunks",
                "destination_collection": "statute_sub_chunks",
                "column": "chunk_text",
                "subsection_column": "sub_chunk_text",
            },
        )
    finally:
        _restore_json5(previous)

    assert response.status_code == 200
    body = response.get_json()
    assert body["llm_errors"] == 2
    assert body["source_rows_skipped"] == 2
    assert body["source_rows_processed"] == 1
    assert body["records_inserted"] == 1
    assert len(inserted) == 1
    assert inserted[0]["subsection_identifier"] == "(a)"
    assert inserted[0]["sub_chunk_text"] == "Alpha line"
    assert inserted[0]["source_id"] == "ok"
    assert stub.raw and stub.raw[0].startswith("{") and stub.raw[0].endswith("}")
    assert "copied" not in inserted[0]["sub_chunk_text"]


def test_statute_subtopics_map_fallback_column_and_coerce_fields() -> None:
    """Empty requested text falls back to chunk_text. Bool and category strings are not normalized."""
    client, coll, anthropic = _app_client()
    inserted: list[dict] = []
    statute = "Must\ndelete\n\ndata."
    coll.find.return_value = [
        {
            "_id": 7,
            "body": "   ",
            "chunk_text": statute,
            "category": " consumer_rights ",
            "jurisdiction": "CA",
        }
    ]
    coll.insert_one.side_effect = lambda doc: inserted.append(doc)
    anthropic.messages.create.return_value = _text_blocks(
        json.dumps(
            {
                "sub_topics": [
                    {
                        "sub_topic": "  right_to_delete  ",
                        "requirement_summary": "  delete data  ",
                        "policy_categories": "nope",
                        "requires_consent": "false",
                        "consumer_facing": 0,
                    },
                    {"requirement_summary": "missing topic"},
                    {"sub_topic": 0, "requirement_summary": "zero"},
                    {
                        "sub_topic": "right_to_access",
                        "policy_categories": [" a ", ""],
                        "requires_consent": False,
                        "consumer_facing": True,
                    },
                ]
            }
        )
    )

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "db",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "body",
            "subsection_column": "statute_text",
            "source_query": {"document_id": "topic-1"},
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["records_inserted"] == 2
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 0
    assert body["llm_errors"] == 0
    assert [row["sub_topic"] for row in inserted] == ["right_to_delete", "right_to_access"]
    first, second = inserted
    assert first["statute_text"] == "Must delete data."
    assert first["chunk_text"] == statute
    assert "body" not in first
    assert first["source_id"] == "7"
    assert first["category"] == "consumer_rights"
    assert first["requirement_summary"] == "delete data"
    assert first["policy_categories"] == []
    assert first["requires_consent"] is True
    assert first["consumer_facing"] is False
    assert second["policy_categories"] == ["a", ""]
    assert second["requires_consent"] is False
    assert second["consumer_facing"] is True
    assert "requirement_summary" not in second
    assert first["jurisdiction"] == "CA"
    prompt = _messages(anthropic)[0]
    assert "pre-classified into category:  consumer_rights " in prompt
    assert "1: Must" in prompt
    coll.find.assert_called_with({"document_id": "topic-1"})
    coll.delete_many.assert_called_with({"document_id": "topic-1"})


def test_statute_subtopics_alias_truncation_and_continue_after_errors() -> None:
    """subtopics is only a fallback, an empty sub_topics list wins, and a long statute is cut only in the prompt."""
    client, coll, anthropic = _app_client()
    inserted: list[dict] = []
    long_text = ("A" * 50000) + "MARKER_END"
    coll.find.return_value = [
        {"_id": "boom", "chunk_text": "will fail"},
        {
            "_id": "both",
            "chunk_text": "both keys",
            "category": "controller_duties",
        },
        {"_id": "alias", "chunk_text": "alias only", "category": ""},
        {"_id": "long", "chunk_text": long_text, "category": "enforcement"},
    ]
    coll.insert_one.side_effect = lambda doc: inserted.append(doc)
    anthropic.messages.create.side_effect = [
        RuntimeError("down"),
        _text_blocks(
            json.dumps(
                {
                    "sub_topics": [],
                    "subtopics": [{"sub_topic": "should_not_win"}],
                }
            )
        ),
        _text_blocks(json.dumps({"subtopics": [{"sub_topic": "privacy_notice"}]})),
        _text_blocks('{"sub_topics":[{"sub_topic":"retention_disclosure",}]}'),
    ]
    stub, previous = _install_json5()
    try:
        response = client.post(
            "/create-statute-subtopics",
            json={
                "database": "db",
                "source_collection": "statute_sub_chunks",
                "destination_collection": "statute_subtopics",
                "column": "chunk_text",
                "subsection_column": "statute_text",
                "parse_prompt": "  extra rule  ",
            },
        )
    finally:
        _restore_json5(previous)

    assert response.status_code == 200
    body = response.get_json()
    assert body["llm_errors"] == 1
    assert body["source_rows_skipped"] == 2
    assert body["source_rows_processed"] == 2
    assert body["records_inserted"] == 2
    by_topic = {row["sub_topic"]: row for row in inserted}
    assert list(by_topic) == ["privacy_notice", "retention_disclosure"]
    assert by_topic["privacy_notice"]["category"] == ""
    assert by_topic["privacy_notice"]["statute_text"] == "alias only"
    assert "requires_consent" not in by_topic["privacy_notice"]
    assert "MARKER_END" in by_topic["retention_disclosure"]["statute_text"]
    prompts = _messages(anthropic)
    assert len(prompts) == 4
    assert all("Additional parsing instructions:\nextra rule" in prompt for prompt in prompts)
    assert "pre-classified into category: controller_duties" in prompts[1]
    assert "pre-classified" not in prompts[2]
    assert "MARKER_END" not in prompts[3]
    assert "A" * 100 in prompts[3]
    assert len(stub.raw) == 1


def test_statute_subtopics_integer_column_does_not_use_fallback_text() -> None:
    """A numeric requested column is stringified. Fallback runs only for missing or blank strings."""
    client, coll, anthropic = _app_client()
    inserted: list[dict] = []
    coll.find.return_value = [
        {
            "_id": "zero",
            "notes": 0,
            "chunk_text": "SHOULD_NOT_FALLBACK",
        }
    ]
    coll.insert_one.side_effect = lambda doc: inserted.append(doc)
    anthropic.messages.create.return_value = _text_blocks(
        json.dumps({"sub_topics": [{"sub_topic": "from_zero"}]})
    )

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "db",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "notes",
            "subsection_column": "statute_text",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["records_inserted"] == 1
    assert inserted[0]["statute_text"] == "0"
    assert inserted[0]["chunk_text"] == "SHOULD_NOT_FALLBACK"
    assert "notes" not in inserted[0]
    prompt = _messages(anthropic)[0]
    assert "1: 0" in prompt
    assert "SHOULD_NOT_FALLBACK" not in prompt


def test_statute_subtopics_insert_error_stops_that_row_only() -> None:
    """A destination write error drops the rest of that statute and still writes the next one."""
    client, coll, anthropic = _app_client()
    seen: list[str] = []

    def insert_one(doc: dict) -> None:
        seen.append(doc["sub_topic"])
        if doc["sub_topic"] == "first":
            raise RuntimeError("disk")

    coll.find.return_value = [
        {"_id": "a", "chunk_text": "one"},
        {"_id": "b", "chunk_text": "two"},
    ]
    coll.insert_one.side_effect = insert_one
    anthropic.messages.create.side_effect = [
        _text_blocks(
            json.dumps({"sub_topics": [{"sub_topic": "first"}, {"sub_topic": "second"}]})
        ),
        _text_blocks(json.dumps({"sub_topics": [{"sub_topic": "third"}]})),
    ]

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "db",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "chunk_text",
            "subsection_column": "statute_text",
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert seen == ["first", "third"]
    assert body["records_inserted"] == 1
    assert body["source_rows_processed"] == 2
    assert body["llm_errors"] == 0
