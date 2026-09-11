"""POST /embedding-models preserves padded whitespace after the non-empty strip check."""
from __future__ import annotations

from unittest.mock import MagicMock

from flask import Flask

from util import init_util, util_bp


def _client() -> tuple:
    wg_db = MagicMock()
    mock_coll = MagicMock()
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), mock_coll


def test_post_embedding_models_preserves_padded_database_and_model_names() -> None:
    """Whitespace that is not empty-after-strip is stored and used as the upsert key as-is."""
    client, mock_coll = _client()
    payload = {
        "database_name": "  privacy-compliance  ",
        "model_name": "  all-MiniLM-L6-v2  ",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine",
            }
        ],
    }

    response = client.post("/embedding-models", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["database_name"] == "  privacy-compliance  "
    assert data["model_name"] == "  all-MiniLM-L6-v2  "

    mock_coll.update_one.assert_called_once()
    filter_doc, update_doc = mock_coll.update_one.call_args[0]
    assert filter_doc == {"database_name": "  privacy-compliance  "}
    assert update_doc["$set"]["database_name"] == "  privacy-compliance  "
    assert update_doc["$set"]["model_name"] == "  all-MiniLM-L6-v2  "
    assert mock_coll.update_one.call_args.kwargs["upsert"] is True


def test_post_embedding_models_preserves_padded_vector_path() -> None:
    """fields[].path is validated with strip() but the original padded path is stored."""
    client, mock_coll = _client()
    payload = {
        "database_name": "privacy-compliance",
        "model_name": "all-MiniLM-L6-v2",
        "fields": [
            {
                "type": "vector",
                "path": "  embedding  ",
                "numDimensions": 384,
                "similarity": "cosine",
            }
        ],
    }

    response = client.post("/embedding-models", json=payload)
    assert response.status_code == 200
    stored_fields = mock_coll.update_one.call_args[0][1]["$set"]["fields"]
    assert stored_fields[0]["path"] == "  embedding  "
