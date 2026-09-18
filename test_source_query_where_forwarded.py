"""Pin current source_query $where forwarding on chunking/parse endpoints.

POST /create-embeddings rejects dangerous operators before find(). Several sibling
write/parse endpoints still pass a dict source_query through to Mongo find() (and
statute subsection/subtopic jobs also delete_many with that same filter). Pin the
current contract so a partial harden cannot silently change only one path.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import core_bp, init_core

WHERE_QUERY = {"$where": "this.jurisdiction == 'CA'"}


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def app(mock_clients) -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def _named_colls(mock_mongo: MagicMock, *names: str) -> Dict[str, MagicMock]:
    """Distinct collection mocks so find vs delete_many can be asserted separately."""
    db = MagicMock()
    colls: Dict[str, MagicMock] = {name: MagicMock() for name in names}

    def _get_coll(name: str) -> MagicMock:
        colls.setdefault(name, MagicMock())
        return colls[name]

    db.__getitem__.side_effect = _get_coll
    mock_mongo.__getitem__.side_effect = lambda _name: db
    mock_mongo.list_database_names.return_value = ["db"]
    db.list_collection_names.return_value = list(names)
    return colls


def test_create_statute_subsections_forwards_where_to_find_and_delete(
    client, mock_clients
) -> None:
    """$where is currently used for both the source read and destination wipe."""
    colls = _named_colls(mock_clients["mongo"], "statute_chunks", "statute_subchunks")
    source = colls["statute_chunks"]
    dest = colls["statute_subchunks"]
    source.find.return_value = []
    dest.delete_many.return_value = MagicMock(deleted_count=0)

    response = client.post(
        "/create-statute-subsections",
        json={
            "database": "db",
            "source_collection": "statute_chunks",
            "destination_collection": "statute_subchunks",
            "column": "chunk_text",
            "subsection_column": "subchunk_text",
            "source_query": WHERE_QUERY,
        },
    )

    assert response.status_code == 200
    source.find.assert_called_once_with(WHERE_QUERY)
    dest.delete_many.assert_called_once_with(WHERE_QUERY)


def test_create_statute_subtopics_forwards_where_to_find_and_delete(
    client, mock_clients
) -> None:
    """Subtopic jobs wipe the destination with the same unvalidated source_query."""
    colls = _named_colls(mock_clients["mongo"], "statute_sub_chunks", "statute_subtopics")
    source = colls["statute_sub_chunks"]
    dest = colls["statute_subtopics"]
    source.find.return_value = []
    dest.delete_many.return_value = MagicMock(deleted_count=0)

    response = client.post(
        "/create-statute-subtopics",
        json={
            "database": "db",
            "source_collection": "statute_sub_chunks",
            "destination_collection": "statute_subtopics",
            "column": "sub_chunk_text",
            "subsection_column": "sub_chunk_text",
            "source_query": WHERE_QUERY,
        },
    )

    assert response.status_code == 200
    source.find.assert_called_once_with(WHERE_QUERY)
    dest.delete_many.assert_called_once_with(WHERE_QUERY)


def test_create_paragraph_sections_forwards_where_to_find_only(
    client, mock_clients
) -> None:
    """Paragraph split currently forwards $where to find() and does not wipe dest."""
    colls = _named_colls(mock_clients["mongo"], "chunks", "sub")
    source = colls["chunks"]
    dest = colls["sub"]
    source.find.return_value = []

    response = client.post(
        "/create-paragraph-sections",
        json={
            "database": "db",
            "source_collection": "chunks",
            "destination_collection": "sub",
            "column": "text",
            "subsection_column": "sub",
            "source_query": WHERE_QUERY,
        },
    )

    assert response.status_code == 200
    source.find.assert_called_once_with(WHERE_QUERY)
    dest.delete_many.assert_not_called()


def test_parse_policy_subsections_forwards_where_to_find(client, mock_clients) -> None:
    """Parse-policy currently forwards $where to find(); empty source skips the LLM."""
    colls = _named_colls(mock_clients["mongo"], "policy_chunks")
    source = colls["policy_chunks"]
    source.find.return_value = []

    response = client.post(
        "/parse-policy-subsections",
        json={
            "database": "db",
            "collection": "policy_chunks",
            "column": "chunk_text",
            "source_query": WHERE_QUERY,
        },
    )

    assert response.status_code == 200
    source.find.assert_called_once_with(WHERE_QUERY)
    payload = response.get_json()
    assert payload["subsections"] == []
    assert payload["source_rows_processed"] == 0
