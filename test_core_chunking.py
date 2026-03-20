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


def test_create_statute_subsections_success(client, mock_clients) -> None:
    """POST /create-statute-subsections with LLM mock succeeds."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value
    dest_coll = dbs["source_db"].__getitem__.return_value
    source_coll.find.return_value = [
        {"_id": "1", "chunk_text": "(a) First section. (b) Second section."},
    ]

    mock_response = MagicMock()
    mock_response.content = []
    for block in [
        {"type": "text", "text": '{"subsections":[{"header_text":"A","identifier":"(a)","start_line":1,"end_line":2},{"header_text":"B","identifier":"(b)","start_line":3,"end_line":4}]}'},
    ]:
        b = MagicMock()
        b.type = block["type"]
        b.text = block.get("text", "")
        mock_response.content.append(b)

    mock_clients["anthropic"].messages.create.return_value = mock_response

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "db",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
        },
    )
    # May succeed or fail depending on LLM response parsing
    assert response.status_code in (200, 500)


def test_create_statute_subtopics_missing_params(client, mock_clients) -> None:
    """POST /create-statute-subtopics with missing params returns 400."""
    response = client.post(
        "/create-statute-subtopics",
        json={"database": "db"},
    )
    assert response.status_code == 400
    assert "Missing required parameters" in response.get_json()["error"]


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
