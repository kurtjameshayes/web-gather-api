"""Leftover POST /embedding-models bool numDimensions acceptance."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from util import init_util, util_bp


@pytest.fixture
def client():
    mock_mongo = MagicMock()
    wg_db = MagicMock()
    mock_coll = MagicMock()
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client(), mock_coll


def test_bool_num_dimensions_is_accepted_because_bool_is_int(client) -> None:
    """True is an int subclass, so numDimensions=True currently passes validation."""
    c, mock_coll = client
    payload = {
        "database_name": "test_db",
        "model_name": "all-MiniLM-L6-v2",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": True,
                "similarity": "cosine",
            }
        ],
    }
    response = c.post("/embedding-models", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["fields"][0]["numDimensions"] is True
    mock_coll.update_one.assert_called_once()


def test_string_num_dimensions_is_rejected(client) -> None:
    """Non-numeric numDimensions is still 400; distinct from bool/float acceptance."""
    c, mock_coll = client
    response = c.post(
        "/embedding-models",
        json={
            "database_name": "test_db",
            "model_name": "all-MiniLM-L6-v2",
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": "384",
                    "similarity": "cosine",
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "numDimensions" in response.get_json()["error"]
    mock_coll.update_one.assert_not_called()
