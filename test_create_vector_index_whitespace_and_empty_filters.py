"""POST /create-vector-index leftover name and empty-filter contracts.

Whitespace database_name/collection_name is 400 before model lookup (#227 only
pins index_name). An explicit empty filter_fields list replaces stored filters
with none; omitting the field still uses embedding_model.filter_fields (#218).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from util import init_util, util_bp


def _client():
    init_util(MagicMock())
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


@pytest.mark.parametrize(
    "payload",
    [
        {"database_name": "   ", "collection_name": "coll"},
        {"database_name": "test_db", "collection_name": "   "},
    ],
)
def test_create_vector_index_whitespace_names_rejected_before_lookup(payload: dict) -> None:
    with patch("util.get_embedding_model_record") as lookup:
        client = _client()
        response = client.post("/create-vector-index", json=payload)
    assert response.status_code == 400
    err = response.get_json()["error"]
    assert "non-empty string" in err
    lookup.assert_not_called()


def test_create_vector_index_empty_filter_fields_replaces_stored() -> None:
    """filter_fields=[] is not None, so stored embedding_model.filter_fields are dropped."""
    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"}
        ],
        "filter_fields": ["document_id", "jurisdiction"],
    }
    target_db = MagicMock()
    target_db.command.side_effect = [{"ok": 1}, {"ok": 1}]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    client = app.test_client()

    with patch("util.get_embedding_model_record", return_value=record):
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "test_db",
                "collection_name": "coll",
                "filter_fields": [],
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["filter_fields_added"] == 0
    filter_fields = [
        f for f in payload["definition"]["fields"] if isinstance(f, dict) and f.get("type") == "filter"
    ]
    assert filter_fields == []
    assert target_db.command.call_count == 2
