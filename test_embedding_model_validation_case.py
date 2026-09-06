"""Embedding-model field validation case and whitespace contracts."""
from __future__ import annotations

from unittest.mock import MagicMock

from flask import Flask

from util import init_util, util_bp


def _client():
    init_util(MagicMock())
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def _payload(fields, database_name="test_db", model_name="all-MiniLM-L6-v2"):
    return {
        "database_name": database_name,
        "model_name": model_name,
        "fields": fields,
    }


def _vector_field(**overrides):
    field = {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 384,
        "similarity": "cosine",
    }
    field.update(overrides)
    return field


def test_embedding_model_type_vector_is_case_sensitive() -> None:
    """type must be exactly 'vector'; 'Vector' is rejected before upsert."""
    client = _client()
    response = client.post(
        "/embedding-models",
        json=_payload([_vector_field(type="Vector")]),
    )
    assert response.status_code == 400
    assert "exactly" in response.get_json()["error"]
    assert "vector" in response.get_json()["error"]


def test_embedding_model_similarity_cosine_is_case_insensitive() -> None:
    """similarity is compared case-insensitively, so COSINE is accepted."""
    mock_mongo = MagicMock()
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    client = app.test_client()

    response = client.post(
        "/embedding-models",
        json=_payload([_vector_field(similarity="COSINE")]),
    )
    assert response.status_code == 200
    mock_mongo.__getitem__.return_value.__getitem__.return_value.update_one.assert_called_once()


def test_embedding_model_whitespace_database_name_rejected() -> None:
    client = _client()
    response = client.post(
        "/embedding-models",
        json=_payload([_vector_field()], database_name="   "),
    )
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]


def test_embedding_model_whitespace_path_rejected() -> None:
    client = _client()
    response = client.post(
        "/embedding-models",
        json=_payload([_vector_field(path="   ")]),
    )
    assert response.status_code == 400
    assert "path" in response.get_json()["error"]


def test_embedding_model_float_num_dimensions_accepted() -> None:
    """numDimensions accepts floats (JSON numbers) as well as ints."""
    mock_mongo = MagicMock()
    init_util(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    client = app.test_client()

    response = client.post(
        "/embedding-models",
        json=_payload([_vector_field(numDimensions=384.0)]),
    )
    assert response.status_code == 200


def test_embedding_model_field_must_be_object() -> None:
    client = _client()
    response = client.post(
        "/embedding-models",
        json=_payload(["embedding"]),
    )
    assert response.status_code == 400
    assert "object" in response.get_json()["error"]
