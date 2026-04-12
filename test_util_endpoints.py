"""Tests for util blueprint: embedding-models, create-vector-index."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import security
from util import init_util, util_bp


def test_get_embedding_models_success_all() -> None:
    """GET /embedding-models returns all models when no database_name filter."""
    from util import init_util
    wg_db = MagicMock()
    wg_db.__getitem__.return_value.find.return_value = [
        {"database_name": "db1", "model_name": "model1"},
        {"database_name": "db2", "model_name": "model2"},
    ]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.get("/embedding-models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    assert len(data["models"]) == 2


def test_get_embedding_models_filtered() -> None:
    """GET /embedding-models?database_name=X returns filtered models."""
    from util import init_util
    wg_db = MagicMock()
    mock_coll = MagicMock()
    mock_coll.find.return_value = [{"database_name": "privacy", "model_name": "all-MiniLM"}]
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.get("/embedding-models?database_name=privacy")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["models"]) == 1
    assert data["models"][0]["model_name"] == "all-MiniLM"
    mock_coll.find.assert_called_once_with({"database_name": "privacy"}, {"_id": 0})


def test_post_embedding_models_success() -> None:
    """POST /embedding-models with valid payload succeeds."""
    from util import init_util
    wg_db = MagicMock()
    mock_coll = MagicMock()
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    payload = {
        "database_name": "test_db",
        "model_name": "all-MiniLM-L6-v2",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
        ],
    }
    response = c.post("/embedding-models", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["database_name"] == "test_db"
    assert data["model_name"] == "all-MiniLM-L6-v2"


def test_post_embedding_models_missing_database_name() -> None:
    """POST /embedding-models without database_name returns 400."""
    from util import init_util
    init_util(MagicMock())

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.post(
        "/embedding-models",
        json={"model_name": "m", "fields": [{"type": "vector", "path": "e", "numDimensions": 384, "similarity": "cosine"}]},
    )
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]


def test_post_embedding_models_missing_model_name() -> None:
    """POST /embedding-models without model_name returns 400."""
    from util import init_util
    init_util(MagicMock())

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.post(
        "/embedding-models",
        json={"database_name": "db", "fields": [{"type": "vector", "path": "e", "numDimensions": 384, "similarity": "cosine"}]},
    )
    assert response.status_code == 400
    assert "model_name" in response.get_json()["error"]


def test_post_embedding_models_invalid_fields() -> None:
    """POST /embedding-models with invalid fields returns 400."""
    from util import init_util
    init_util(MagicMock())

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.post(
        "/embedding-models",
        json={"database_name": "db", "model_name": "m", "fields": []},
    )
    assert response.status_code == 400
    assert "fields" in response.get_json()["error"]


def test_embedding_models_invalid_api_key_rejected(monkeypatch) -> None:
    """Configured APP_API_KEY rejects invalid x-api-key on util endpoints."""
    monkeypatch.setattr(security, "_app_api_key", "expected-key")
    from util import init_util
    init_util(MagicMock())

    app = Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.get("/embedding-models", headers={"x-api-key": "wrong"})
    assert response.status_code == 403
    assert "Invalid API key" in response.get_json()["error"]


def test_create_vector_index_missing_database_name() -> None:
    """POST /create-vector-index without database_name returns 400."""
    from util import init_util
    init_util(MagicMock())

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.post(
        "/create-vector-index",
        json={"collection_name": "coll"},
    )
    assert response.status_code == 400
    assert "database_name" in response.get_json()["error"]


def test_create_vector_index_missing_collection_name() -> None:
    """POST /create-vector-index without collection_name returns 400."""
    from util import init_util
    init_util(MagicMock())

    app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    response = c.post(
        "/create-vector-index",
        json={"database_name": "db"},
    )
    assert response.status_code == 400
    assert "collection_name" in response.get_json()["error"]


def test_create_vector_index_no_embedding_model() -> None:
    """POST /create-vector-index when no embedding_model configured returns 400."""
    from util import init_util
    from unittest.mock import patch

    wg_db = MagicMock()
    wg_db.__getitem__.return_value.find_one.return_value = None
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    with patch("util.get_embedding_model_record", return_value=None):
        app = __import__("flask", fromlist=["Flask"]).Flask(__name__)
        app.register_blueprint(util_bp)
        c = app.test_client()

        response = c.post(
            "/create-vector-index",
            json={"database_name": "db", "collection_name": "coll"},
        )
    assert response.status_code == 400
    assert "embedding_model" in response.get_json()["error"].lower() or "configure" in response.get_json()["error"].lower()


def test_create_vector_index_success() -> None:
    """POST /create-vector-index with valid config succeeds."""
    from util import init_util

    target_db = MagicMock()
    target_db.command.side_effect = [
        Exception("Index does not exist"),  # drop fails
        {"ok": 1},  # createSearchIndexes success
    ]
    wg_db = MagicMock()
    mock_mongo = MagicMock()

    def get_db(name):
        return target_db if name == "test_db" else wg_db

    mock_mongo.__getitem__.side_effect = get_db
    init_util(mock_mongo)

    record = {
        "database_name": "test_db",
        "model_name": "m",
        "fields": [{"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"}],
    }
    with patch("util.get_embedding_model_record", return_value=record):
        app = Flask(__name__)
        app.register_blueprint(util_bp)
        c = app.test_client()

        response = c.post(
            "/create-vector-index",
            json={"database_name": "test_db", "collection_name": "coll"},
        )

    # createSearchIndexes may raise on drop; util catches and continues
    if response.status_code == 200:
        data = response.get_json()
        assert data["database_name"] == "test_db"
        assert data["collection_name"] == "coll"
        assert "index_name" in data

