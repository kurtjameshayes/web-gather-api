"""POST /embedding-models upsert write contract.

Validator shape checks live in other coverage PRs. These tests pin the Mongo
write: one document per database_name, $set of the payload plus updated_at,
and upsert=True so a missing record is created rather than silently dropped.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from db import EMBEDDING_MODEL_COLLECTION, WEB_GATHER_DB
from util import init_util, util_bp


FIXED_NOW = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)

VALID_PAYLOAD = {
    "database_name": "privacy-compliance",
    "model_name": "all-MiniLM-L6-v2",
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine",
        }
    ],
}


@pytest.fixture
def coll() -> MagicMock:
    mock_coll = MagicMock()
    wg_db = MagicMock()
    wg_db.__getitem__.return_value = mock_coll
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)
    return mock_coll


@pytest.fixture
def client(coll: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(util_bp)
    return app.test_client()


def test_post_embedding_models_upserts_by_database_name(client, coll) -> None:
    """Writes are keyed only by database_name so a later POST replaces the model."""
    with patch("util.utc_now", return_value=FIXED_NOW):
        response = client.post("/embedding-models", json=VALID_PAYLOAD)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["database_name"] == "privacy-compliance"
    assert payload["model_name"] == "all-MiniLM-L6-v2"
    assert "updated_at" in payload

    coll.update_one.assert_called_once()
    filter_doc, update_doc = coll.update_one.call_args.args[:2]
    kwargs = coll.update_one.call_args.kwargs
    assert filter_doc == {"database_name": "privacy-compliance"}
    assert kwargs.get("upsert") is True
    set_doc = update_doc["$set"]
    assert set_doc["database_name"] == "privacy-compliance"
    assert set_doc["model_name"] == "all-MiniLM-L6-v2"
    assert set_doc["fields"] == VALID_PAYLOAD["fields"]
    assert set_doc["updated_at"] == FIXED_NOW


def test_post_embedding_models_targets_web_gather_embedding_model_collection(client, coll) -> None:
    """Config must land in web-gather.embedding_model, not the target application DB."""
    mock_mongo = MagicMock()
    wg_db = MagicMock()
    wg_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = wg_db
    init_util(mock_mongo)

    app = Flask(__name__)
    app.register_blueprint(util_bp)
    c = app.test_client()

    with patch("util.utc_now", return_value=FIXED_NOW):
        response = c.post("/embedding-models", json=VALID_PAYLOAD)

    assert response.status_code == 200
    mock_mongo.__getitem__.assert_called_with(WEB_GATHER_DB)
    wg_db.__getitem__.assert_called_with(EMBEDDING_MODEL_COLLECTION)
