"""POST /create-embeddings rejects a list-typed source_query before Mongo.

A JSON-string list (\"[1, 2]\") is 400 after json.loads. A list-typed body
value is 400 in the else-branch before json.loads. Distinct from statute
subsection/subtopic/paragraph jobs, which coerce non-dicts to {}.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from core import core_bp, init_core
from conftest import _wire_mongo_core


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


@pytest.fixture
def client(mock_clients):
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_create_embeddings_list_source_query_is_400_before_find(
    client, mock_clients
) -> None:
    """List-typed source_query is 400 'must be a JSON object'; find() is not called."""
    dbs = _wire_mongo_core(mock_clients["mongo"])
    source_coll = dbs["source_db"].__getitem__.return_value

    response = client.post(
        "/create-embeddings",
        json={
            "source_database_name": "src",
            "source_collection_name": "docs",
            "index_database_name": "index_db",
            "index_collection_name": "chunks",
            "source_query": [1, 2],
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "source_query must be a JSON object"
    source_coll.find.assert_not_called()


def test_create_embeddings_list_source_query_before_required_fields(
    client, mock_clients
) -> None:
    """List-typed source_query is rejected before the missing-params check."""
    response = client.post("/create-embeddings", json={"source_query": ["x"]})
    assert response.status_code == 400
    err = response.get_json()["error"]
    assert err == "source_query must be a JSON object"
    assert "Missing required parameters" not in err
