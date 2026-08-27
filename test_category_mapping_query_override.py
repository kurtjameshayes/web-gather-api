"""GET /category-mapping query param replaces simple filters rather than merging."""
from __future__ import annotations

import urllib.parse

from bson import ObjectId
from flask import Flask
from unittest.mock import MagicMock

import pytest

from db import db_bp, init_db


@pytest.fixture
def mock_mongo():
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    return mock_mongo


@pytest.fixture
def mock_coll(mock_mongo):
    pc_db = MagicMock()
    mock_coll = MagicMock()
    pc_db.__getitem__.return_value = mock_coll
    mock_mongo.__getitem__.return_value = pc_db
    return mock_coll


@pytest.fixture
def app(mock_mongo):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_query_param_replaces_statute_category_filter(client, mock_coll) -> None:
    """query replaces statute_category instead of AND-merging the two filters."""
    mock_coll.find.return_value = []
    query = urllib.parse.quote('{"sub_topic": "right_to_delete"}')
    response = client.get(
        f"/category-mapping?statute_category=retention&query={query}"
    )
    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({"sub_topic": "right_to_delete"})


def test_query_param_replaces_sub_topic_filter(client, mock_coll) -> None:
    """query replaces sub_topic instead of AND-merging the two filters."""
    mock_coll.find.return_value = []
    query = urllib.parse.quote('{"statute_category": "enforcement"}')
    response = client.get(f"/category-mapping?sub_topic=right_to_know&query={query}")
    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({"statute_category": "enforcement"})


def test_query_param_alone_is_used_as_find_filter(client, mock_coll) -> None:
    """A valid query object is passed through to find when no simple filters are set."""
    oid = ObjectId()
    mock_coll.find.return_value = [
        {"_id": oid, "statute_category": "retention", "policy_categories": ["data_retention"]},
    ]
    query = urllib.parse.quote('{"statute_category": "retention"}')
    response = client.get(f"/category-mapping?query={query}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["category_mappings"][0]["_id"] == str(oid)
    mock_coll.find.assert_called_once_with({"statute_category": "retention"})
