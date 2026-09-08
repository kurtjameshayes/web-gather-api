"""Regression tests for leftover /create-paragraph-sections source_query coercion.

Invalid JSON is 400. A JSON array, empty string, or other non-dict value is
silently coerced to {} and the source find() is unscoped — processing every
row instead of the intended subset.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())
sys.modules.setdefault("pypdf", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def mock_clients() -> dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients: dict[str, Any]):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _wire_source_and_dest(mock_mongo: MagicMock) -> tuple[MagicMock, MagicMock]:
    source_coll = MagicMock()
    dest_coll = MagicMock()
    source_coll.find.return_value = [
        {"_id": "1", "chunk_text": "Only one paragraph."},
    ]
    dest_coll.insert_one.return_value = MagicMock()

    db = MagicMock()

    def _get_coll(name: str) -> MagicMock:
        if name == "chunks":
            return source_coll
        return dest_coll

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.return_value = db
    return source_coll, dest_coll


_BASE_BODY = {
    "database": "db",
    "source_collection": "chunks",
    "destination_collection": "subchunks",
    "column": "chunk_text",
    "subsection_column": "subchunk_text",
}


def test_create_paragraph_sections_list_source_query_is_unscoped(
    client, mock_clients
) -> None:
    """A JSON-array source_query is coerced to {} rather than rejected."""
    source_coll, _dest = _wire_source_and_dest(mock_clients["mongo"])
    response = client.post(
        "/create-paragraph-sections",
        json={**_BASE_BODY, "source_query": ["document_id", "x"]},
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})
    data = response.get_json()
    assert data["records_inserted"] == 1
    assert data["source_rows_processed"] == 1


def test_create_paragraph_sections_json_string_array_is_unscoped(
    client, mock_clients
) -> None:
    """A JSON-string array source_query parses then falls back to unscoped {}."""
    source_coll, _dest = _wire_source_and_dest(mock_clients["mongo"])
    response = client.post(
        "/create-paragraph-sections",
        json={**_BASE_BODY, "source_query": '["document_id"]'},
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})


def test_create_paragraph_sections_empty_string_source_query_is_unscoped(
    client, mock_clients
) -> None:
    """Empty-string source_query skips JSON parse and keeps the default {}."""
    source_coll, _dest = _wire_source_and_dest(mock_clients["mongo"])
    response = client.post(
        "/create-paragraph-sections",
        json={**_BASE_BODY, "source_query": ""},
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with({})


def test_create_paragraph_sections_dict_source_query_is_applied(
    client, mock_clients
) -> None:
    """A dict source_query is passed through to find() unchanged."""
    source_coll, _dest = _wire_source_and_dest(mock_clients["mongo"])
    query = {"document_id": "stat-1"}
    response = client.post(
        "/create-paragraph-sections",
        json={**_BASE_BODY, "source_query": query},
    )
    assert response.status_code == 200
    source_coll.find.assert_called_once_with(query)
