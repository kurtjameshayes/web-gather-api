"""Field coercion for POST /category-mapping.

Stored category strings are what v4 gap analysis matches. Falsy statute
categories are rejected, while a numeric sub-topic is kept.
"""
from __future__ import annotations

from bson import ObjectId
from flask import Flask
from unittest.mock import MagicMock

from db import CATEGORY_MAPPING_COLLECTION, PRIVACY_COMPLIANCE_DB, db_bp, init_db


def _client():
    mongo = MagicMock()
    init_db(mongo)
    coll = MagicMock()
    coll.insert_one.return_value.inserted_id = ObjectId()
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client(), coll


def test_mapping_fields_are_stripped_and_stringified() -> None:
    client, coll = _client()
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "  consumer_rights  ",
            "policy_categories": ["  sale  ", 0, False],
            "sub_topic": 0,
            "description": "  note  ",
        },
    )
    assert response.status_code == 201
    assert response.get_json()["database"] == PRIVACY_COMPLIANCE_DB
    assert response.get_json()["collection"] == CATEGORY_MAPPING_COLLECTION
    stored = coll.insert_one.call_args[0][0]
    assert stored["statute_category"] == "consumer_rights"
    assert stored["policy_categories"] == ["sale", "0", "False"]
    assert stored["sub_topic"] == "0"
    assert stored["description"] == "note"


def test_numeric_zero_statute_category_is_rejected() -> None:
    client, coll = _client()
    response = client.post(
        "/category-mapping",
        json={"statute_category": 0, "policy_categories": ["sale"], "sub_topic": 0},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "statute_category is required"
    coll.insert_one.assert_not_called()


def test_omitted_optional_fields_are_absent() -> None:
    client, coll = _client()
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "controller_duties",
            "policy_categories": ["security"],
        },
    )
    assert response.status_code == 201
    stored = coll.insert_one.call_args[0][0]
    assert "sub_topic" not in stored
    assert "description" not in stored
