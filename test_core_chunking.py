"""Tests for core chunking endpoints: create-paragraph-sections, create-statute-subsections,
create-statute-subtopics, parse-policy-subsections, create-chunks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from core import WEB_GATHER_DB, core_bp, init_core
from conftest import _wire_mongo_core


@pytest.fixture
def mock_clients():
    """Mock MongoDB, Firecrawl, Anthropic for core."""
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {"mongo": mock_mongo, "firecrawl": mock_firecrawl, "anthropic": mock_anthropic}


@pytest.fixture
def app(mock_clients):
    """Flask app with core blueprint."""
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    """Test client."""
    return app.test_client()


def test_create_paragraph_sections_success(client, mock_clients) -> None:
    """POST /create-paragraph-sections splits by paragraphs and writes to dest."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    dest_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "1", "chunk_text": "Para one.\n\nPara two.\n\nPara three."},
    ]

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "db",
            "source_collection": "chunks",
            "destination_collection": "subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 3
    assert data["source_rows_processed"] == 1
    assert data["source_collection"] == "chunks"
    assert data["destination_collection"] == "subchunks"


def test_create_paragraph_sections_missing_params(client, mock_clients) -> None:
    """POST /create-paragraph-sections with missing params returns 400."""
    response = client.post(
        "/create-paragraph-sections",
        json={"database": "db", "source_collection": "chunks"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_create_paragraph_sections_invalid_source_query(client, mock_clients) -> None:
    """POST /create-paragraph-sections with invalid source_query JSON returns 400."""
    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "db",
            "source_collection": "chunks",
            "destination_collection": "sub",
            "column": "text",
            "subsection_column": "sub",
            "source_query": "{bad}",
        },
    )
    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]


def test_create_statute_subsections_missing_params(client, mock_clients) -> None:
    """POST /create-statute-subsections with missing params returns 400."""
    response = client.post(
        "/create-statute-subsections",
        json={"database": "db"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def _llm_text_response(text: str) -> MagicMock:
    """Build an Anthropic-like messages.create response with one text block."""
    mock_response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    mock_response.content = [block]
    return mock_response


def test_create_statute_subsections_success_extracts_line_ranges_and_metadata(
    client, mock_clients
) -> None:
    """Statute subsections should use line ranges, prepend chunk headers, and persist metadata."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    deleted = MagicMock()
    deleted.deleted_count = 2
    coll.delete_many.return_value = deleted
    coll.find.return_value = [
        {
            "_id": "src-1",
            "document_id": "stat-1",
            "chunk_text": (
                "# 1798.105. Right to Delete\n"
                "(a) A consumer may request deletion.\n"
                "(b) A business shall comply within 45 days."
            ),
        },
    ]

    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"subsections":['
        '{"header_text":"Right to request deletion","identifier":"(a)",'
        '"category":"consumer_rights","category_reasoning":"Creates a consumer right.",'
        '"start_line":2,"end_line":2},'
        '{"header_text":"Business duty","identifier":"(b)",'
        '"category":"controller_duties","start_line":3,"end_line":3}'
        "]}"
    )

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_sub_chunks",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": {"document_id": "stat-1"},
            "parse_prompt": "Keep nested numeric markers together.",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 2
    assert data["source_rows_processed"] == 1
    assert data["source_rows_skipped"] == 0
    assert data["llm_errors"] == 0

    coll.find.assert_called_once_with({"document_id": "stat-1"})
    coll.delete_many.assert_called_once_with({"document_id": "stat-1"})
    assert coll.insert_one.call_count == 2

    first = coll.insert_one.call_args_list[0].args[0]
    second = coll.insert_one.call_args_list[1].args[0]
    assert first["subsection_identifier"] == "(a)"
    assert first["header_text"] == "Right to request deletion"
    assert first["category"] == "consumer_rights"
    assert first["category_reasoning"] == "Creates a consumer right."
    assert first["source_id"] == "src-1"
    assert first["document_id"] == "stat-1"
    assert first["sub_chunk_text"].startswith("# 1798.105. Right to Delete")
    assert "(a) A consumer may request deletion." in first["sub_chunk_text"]
    assert second["subsection_identifier"] == "(b)"
    assert second["category"] == "controller_duties"
    assert "(b) A business shall comply within 45 days." in second["sub_chunk_text"]

    prompt = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Additional parsing instructions:" in prompt
    assert "Keep nested numeric markers together." in prompt
    assert "1: # 1798.105. Right to Delete" in prompt


def test_create_statute_subsections_llm_failure_counts_error_without_failing_request(
    client, mock_clients
) -> None:
    """LLM failures for a source row should increment llm_errors and still return 200."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.delete_many.return_value = MagicMock(deleted_count=0)
    coll.find.return_value = [
        {"_id": "1", "chunk_text": "(a) A consumer may request deletion."},
        {"_id": "2", "chunk_text": "   "},
    ]
    mock_clients["anthropic"].messages.create.side_effect = RuntimeError("anthropic down")

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
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 0
    assert data["llm_errors"] == 1
    assert data["source_rows_skipped"] == 2
    assert data["source_rows_processed"] == 0
    coll.insert_one.assert_not_called()


def test_create_statute_subsections_invalid_json_and_source_query(
    client, mock_clients
) -> None:
    """Invalid source_query JSON is 400; unparseable LLM JSON increments llm_errors."""
    bad_query = client.post(
        "/create-statute-subsections",
        json={
            "database": "db",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_sub_chunks",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": "{bad",
        },
    )
    assert bad_query.status_code == 400
    assert "source_query" in bad_query.get_json()["error"]

    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.delete_many.return_value = MagicMock(deleted_count=0)
    coll.find.return_value = [{"_id": "1", "chunk_text": "(a) Text."}]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        "Sorry, I cannot comply."
    )

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
    assert response.status_code == 200
    data = response.get_json()
    assert data["llm_errors"] == 1
    assert data["records_inserted"] == 0
    coll.insert_one.assert_not_called()


def test_create_statute_subtopics_missing_params(client, mock_clients) -> None:
    """POST /create-statute-subtopics with missing params returns 400."""
    response = client.post(
        "/create-statute-subtopics",
        json={"database": "db"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_create_statute_subtopics_success_uses_fallback_column_and_metadata(
    client, mock_clients
) -> None:
    """Subtopics should fall back to alternate text columns and persist LLM metadata."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.delete_many.return_value = MagicMock(deleted_count=1)
    coll.find.return_value = [
        {
            "_id": "sub-1",
            "document_id": "stat-9",
            "category": "consumer_rights",
            "chunk_text": "",
            "sub_chunk_text": "A consumer may request deletion of personal information.",
        }
    ]
    mock_clients["anthropic"].messages.create.return_value = _llm_text_response(
        '{"sub_topics":[{'
        '"sub_topic":"Right to delete",'
        '"requirement_summary":"Consumers can demand deletion.",'
        '"policy_categories":["deletion","retention"],'
        '"requires_consent":false,'
        '"consumer_facing":true'
        "}]}"
    )

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "privacy-compliance",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": {"document_id": "stat-9"},
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1
    assert data["llm_errors"] == 0

    coll.delete_many.assert_called_once_with({"document_id": "stat-9"})
    inserted = coll.insert_one.call_args.args[0]
    assert inserted["sub_topic"] == "Right to delete"
    assert inserted["requirement_summary"] == "Consumers can demand deletion."
    assert inserted["policy_categories"] == ["deletion", "retention"]
    assert inserted["requires_consent"] is False
    assert inserted["consumer_facing"] is True
    assert inserted["category"] == "consumer_rights"
    assert inserted["source_id"] == "sub-1"
    assert inserted["sub_chunk_text"] == (
        "A consumer may request deletion of personal information."
    )

    prompt = mock_clients["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
    assert "pre-classified into category: consumer_rights" in prompt
    assert "A consumer may request deletion of personal information." in prompt


def test_create_statute_subtopics_llm_failure_is_non_fatal(client, mock_clients) -> None:
    """Subtopic LLM failures should be counted without failing the HTTP request."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    coll = dbs["source_db"].__getitem__.return_value
    coll.delete_many.return_value = MagicMock(deleted_count=0)
    coll.find.return_value = [
        {"_id": "1", "sub_chunk_text": "Controllers must provide a privacy notice."}
    ]
    mock_clients["anthropic"].messages.create.side_effect = RuntimeError("timeout")

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "db",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "sub_chunk_text",
            "subsection_column": "sub_chunk_text",
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["llm_errors"] == 1
    assert data["records_inserted"] == 0
    assert data["source_rows_skipped"] == 1
    coll.insert_one.assert_not_called()


def test_parse_policy_subsections_missing_params(client, mock_clients) -> None:
    """POST /parse-policy-subsections with missing params returns 400."""
    response = client.post(
        "/parse-policy-subsections",
        json={"database": "db"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_parse_policy_subsections_success(client, mock_clients) -> None:
    """POST /parse-policy-subsections returns subsections from LLM."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "1", "text": "Section one. Section two."},
    ]

    mock_response = MagicMock()
    b = MagicMock()
    b.type = "text"
    b.text = '{"subsections":[{"subsection_identifier":"1","subsection_text":"Section one.","heading":"Intro"},{"subsection_identifier":"2","subsection_text":"Section two."}]}'
    mock_response.content = [b]
    mock_clients["anthropic"].messages.create.return_value = mock_response

    with patch("core._parse_policy_section_to_subsections") as mock_parse:
        mock_parse.return_value = (
            [
                {"subsection_identifier": "1", "subsection_text": "Section one.", "heading": "Intro"},
                {"subsection_identifier": "2", "subsection_text": "Section two."},
            ],
            None,
        )
        response = client.post(
            "/parse-policy-subsections",
            json={
                "database": "db",
                "collection": "policy_chunks",
                "column": "text",
            },
        )
    assert response.status_code == 200
    data = response.get_json()
    assert "subsections" in data
    assert data["column"] == "text"


def test_create_chunks_success(client, mock_clients) -> None:
    """POST /create-chunks chunks text and writes to destination."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    dest_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "1", "text": "This is a longer piece of text that will be chunked into smaller parts for embedding."},
    ]

    response = client.post(
        "/create-chunks",
        json={
            "database": "db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 50,
            "overlap": 10,
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] >= 1
    assert data["source_collection"] == "docs"
    assert data["destination_collection"] == "chunks"
    assert data["chunk_size"] == 50
    assert data["overlap"] == 10


def test_create_chunks_missing_params(client, mock_clients) -> None:
    """POST /create-chunks with missing params returns 400."""
    response = client.post(
        "/create-chunks",
        json={"database": "db", "source_collection": "docs"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


def test_create_chunks_invalid_chunk_size(client, mock_clients) -> None:
    """POST /create-chunks with non-integer chunk_size returns 400."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [{"_id": "1", "text": "x"}]

    response = client.post(
        "/create-chunks",
        json={
            "database": "db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": "not-a-number",
            "overlap": 0,
        },
    )
    assert response.status_code == 400
    assert "chunk_size" in response.get_json()["error"].lower() or "integer" in response.get_json()["error"].lower()
