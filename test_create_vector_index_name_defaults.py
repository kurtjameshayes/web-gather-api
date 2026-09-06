"""POST /create-vector-index index_name default vs whitespace rejection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from flask import Flask

from util import init_util, util_bp


def _record():
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "euclidean",
            }
        ]
    }


def _client_with_mongo():
    mock_mongo = MagicMock()
    mock_db = MagicMock()
    mock_mongo.__getitem__.return_value = mock_db
    mock_db.command.side_effect = [Exception("index missing"), {"ok": 1}]
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), mock_db


def _body(**overrides):
    payload = {
        "database_name": "privacy-compliance",
        "collection_name": "policy_legal_embeddings",
    }
    payload.update(overrides)
    return payload


def test_create_vector_index_omitted_index_name_defaults() -> None:
    """Omitted index_name becomes vector_index on the Atlas create command."""
    client, mock_db = _client_with_mongo()
    with patch("util.get_embedding_model_record", return_value=_record()):
        response = client.post("/create-vector-index", json=_body())
    assert response.status_code == 200
    data = response.get_json()
    assert data["index_name"] == "vector_index"
    create_cmd = mock_db.command.call_args_list[-1][0][0]
    assert create_cmd["indexes"][0]["name"] == "vector_index"


def test_create_vector_index_empty_string_index_name_defaults() -> None:
    """Empty string is falsy and is replaced with vector_index before validation."""
    client, mock_db = _client_with_mongo()
    with patch("util.get_embedding_model_record", return_value=_record()):
        response = client.post("/create-vector-index", json=_body(index_name=""))
    assert response.status_code == 200
    assert response.get_json()["index_name"] == "vector_index"
    create_cmd = mock_db.command.call_args_list[-1][0][0]
    assert create_cmd["indexes"][0]["name"] == "vector_index"


def test_create_vector_index_whitespace_index_name_rejected() -> None:
    """Whitespace-only index_name is truthy, then fails the strip() non-empty check."""
    client, mock_db = _client_with_mongo()
    with patch("util.get_embedding_model_record", return_value=_record()) as lookup:
        response = client.post("/create-vector-index", json=_body(index_name="   "))
    assert response.status_code == 400
    assert "index_name" in response.get_json()["error"]
    lookup.assert_not_called()
    mock_db.command.assert_not_called()
