"""Regression tests for DELETE /category-mapping query contracts.

Empty `{}` is a valid parsed query, so it currently wipes the whole mapping
collection. Category mappings drive v4 statute-policy pairing; an unscoped
delete is a high-blast-radius data-loss path that base tests never pin.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock
from urllib.parse import quote

from bson import ObjectId
from flask import Flask

from db import db_bp, init_db


def _client(mock_coll: MagicMock):
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    init_db(mock_mongo)
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client(), mock_coll


def test_delete_category_mapping_empty_object_wipes_collection() -> None:
    """DELETE /category-mapping?query={} currently calls delete_many({})."""
    mock_coll = MagicMock()
    mock_coll.delete_many.return_value.deleted_count = 12
    client, coll = _client(mock_coll)

    response = client.delete("/category-mapping?query=%7B%7D")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted_count"] == 12
    coll.delete_many.assert_called_once_with({})


def test_delete_category_mapping_query_converts_oid() -> None:
    """DELETE /category-mapping converts a valid Extended JSON $oid to ObjectId."""
    mock_coll = MagicMock()
    mock_coll.delete_many.return_value.deleted_count = 1
    client, coll = _client(mock_coll)
    oid_hex = "64b8f7b7b1f1eaf5b2f0c001"
    query = quote(json.dumps({"_id": {"$oid": oid_hex}}))

    response = client.delete(f"/category-mapping?query={query}")

    assert response.status_code == 200
    coll.delete_many.assert_called_once()
    passed = coll.delete_many.call_args[0][0]
    assert passed["_id"] == ObjectId(oid_hex)


def test_delete_category_mapping_invalid_query_json_does_not_delete() -> None:
    """Malformed query JSON is 400 and must not call delete_many."""
    mock_coll = MagicMock()
    client, coll = _client(mock_coll)

    response = client.delete("/category-mapping?query=not-json")

    assert response.status_code == 400
    assert "json" in response.get_json()["error"].lower()
    coll.delete_many.assert_not_called()
