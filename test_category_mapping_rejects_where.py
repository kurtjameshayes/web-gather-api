"""GET/DELETE /category-mapping parse query via _parse_safe_query.

Unlike statute source_query paths that currently forward $where (#239),
category-mapping rejects $where/$regex before find/delete_many. Documents
already pin this helper; these tests pin the mapping endpoints that drive
v4 statute-policy pairing.
"""
from __future__ import annotations

import urllib.parse
from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def mock_coll():
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    pc_db = MagicMock()
    coll = MagicMock()
    pc_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = pc_db
    return coll


@pytest.fixture
def client(mock_coll):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_get_category_mapping_rejects_where(client, mock_coll) -> None:
    """GET /category-mapping?query={$where:...} is 400 and does not call find()."""
    query = urllib.parse.quote('{"$where": "1==1"}')
    response = client.get(f"/category-mapping?query={query}")
    assert response.status_code == 400
    assert "$where" in response.get_json()["error"]
    mock_coll.find.assert_not_called()


def test_delete_category_mapping_rejects_where(client, mock_coll) -> None:
    """DELETE /category-mapping?query={$where:...} is 400 and does not delete."""
    query = urllib.parse.quote('{"$where": "this.statute_category == 1"}')
    response = client.delete(f"/category-mapping?query={query}")
    assert response.status_code == 400
    assert "$where" in response.get_json()["error"]
    mock_coll.delete_many.assert_not_called()


def test_get_category_mapping_rejects_regex(client, mock_coll) -> None:
    """$regex is also blocked (same operator denylist as GET /documents)."""
    query = urllib.parse.quote('{"statute_category": {"$regex": ".*"}}')
    response = client.get(f"/category-mapping?query={query}")
    assert response.status_code == 400
    assert "$regex" in response.get_json()["error"]
    mock_coll.find.assert_not_called()
