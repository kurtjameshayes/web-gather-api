"""Regression tests for POST /create-statute-subsections header prepend and empty-column skip.

Subsection rows prepend the source chunk's `#` header when the extracted text
does not already start with `#`. Unlike create-statute-subtopics, an empty
requested column does **not** fall back to another text field.
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


def _post_subsections(client, **extra: Any):
    payload = {
        "database": "privacy-compliance",
        "source_collection": "src",
        "destination_collection": "dest",
        "column": "chunk_text",
        "subsection_column": "sub_chunk_text",
    }
    payload.update(extra)
    return client.post("/create-statute-subsections", json=payload)


def test_hash_header_is_prepended_when_extracted_text_lacks_it(client, mock_clients) -> None:
    """A leading `#` source line is prepended so downstream chunks keep the citation header."""
    source_text = (
        "# 1798.105. Consumers' Right to Delete Personal Information\n"
        "(a) A consumer shall have the right to delete personal information."
    )
    _source_coll, dest_coll = _wire_source_dest(
        mock_clients["mongo"],
        [{"_id": "sec-1", "chunk_text": source_text, "jurisdiction": "CA"}],
    )
    mock_clients["anthropic"].messages.create.return_value = _llm_json(
        {
            "subsections": [
                {
                    "identifier": "(a)",
                    "header_text": "Right to delete",
                    "category": "consumer_rights",
                    "category_reasoning": "Establishes a deletion right.",
                    "start_line": 2,
                    "end_line": 2,
                }
            ]
        }
    )

    response = _post_subsections(client)
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1
    assert data["llm_errors"] == 0

    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["subsection_identifier"] == "(a)"
    assert inserted["sub_chunk_text"].startswith(
        "# 1798.105. Consumers' Right to Delete Personal Information"
    )
    assert "A consumer shall have the right to delete personal information." in inserted["sub_chunk_text"]
    # Destination keeps the original split column (unlike create-statute-subtopics).
    assert inserted["chunk_text"] == source_text
    assert inserted["header_text"] == "Right to delete"
    assert inserted["category"] == "consumer_rights"
    assert inserted["category_reasoning"] == "Establishes a deletion right."
    assert inserted["jurisdiction"] == "CA"
    assert inserted["source_id"] == "sec-1"


def test_header_not_prepended_when_extracted_text_already_starts_with_hash(client, mock_clients) -> None:
    """Do not double-prefix when the extracted subsection already starts with `#`."""
    source_text = "# 1798.105. Title\n# (a) already headed subsection text"
    _source_coll, dest_coll = _wire_source_dest(
        mock_clients["mongo"],
        [{"_id": "sec-2", "chunk_text": source_text}],
    )
    mock_clients["anthropic"].messages.create.return_value = _llm_json(
        {
            "subsections": [
                {
                    "identifier": "(a)",
                    "start_line": 2,
                    "end_line": 2,
                }
            ]
        }
    )

    response = _post_subsections(client)
    assert response.status_code == 200
    inserted = dest_coll.insert_one.call_args[0][0]
    assert inserted["sub_chunk_text"] == "# (a) already headed subsection text"
    assert inserted["sub_chunk_text"].count("# 1798.105. Title") == 0


def test_empty_requested_column_does_not_fall_back(client, mock_clients) -> None:
    """Unlike create-statute-subtopics, empty `column` is a skip — no other field is used."""
    _wire_source_dest(
        mock_clients["mongo"],
        [
            {
                "_id": "sec-3",
                "chunk_text": "   ",
                "sub_chunk_text": "MUST_NOT_BE_SENT_TO_LLM",
                "text": "ALSO_MUST_NOT_BE_SENT",
            }
        ],
    )

    response = _post_subsections(client)
    assert response.status_code == 200
    data = response.get_json()
    assert data["records_inserted"] == 0
    assert data["source_rows_skipped"] == 1
    assert data["source_rows_processed"] == 0
    mock_clients["anthropic"].messages.create.assert_not_called()
