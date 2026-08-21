"""Regression tests for POST /category-mapping insert payload shape."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from bson import ObjectId
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
    coll.insert_one.return_value.inserted_id = ObjectId()
    return coll


@pytest.fixture
def client(mock_coll):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_post_category_mapping_strips_fields_and_includes_optional(client, mock_coll) -> None:
    """Inserted mapping must strip whitespace and persist optional fields."""
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "  retention  ",
            "policy_categories": [" data_retention ", 123],
            "sub_topic": " right_to_delete ",
            "description": "  keep for 12 months  ",
        },
    )

    assert response.status_code == 201
    inserted = mock_coll.insert_one.call_args[0][0]
    assert inserted["statute_category"] == "retention"
    assert inserted["policy_categories"] == ["data_retention", "123"]
    assert inserted["sub_topic"] == "right_to_delete"
    assert inserted["description"] == "keep for 12 months"


def test_post_category_mapping_omits_optional_fields_when_absent(client, mock_coll) -> None:
    """Optional sub_topic/description must not be written when omitted."""
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "consent",
            "policy_categories": ["opt_in"],
        },
    )

    assert response.status_code == 201
    inserted = mock_coll.insert_one.call_args[0][0]
    assert inserted["statute_category"] == "consent"
    assert inserted["policy_categories"] == ["opt_in"]
    assert "sub_topic" not in inserted
    assert "description" not in inserted


def test_post_category_mapping_rejects_empty_policy_categories(client, mock_coll) -> None:
    """Empty policy_categories is treated as missing and must not insert."""
    response = client.post(
        "/category-mapping",
        json={"statute_category": "retention", "policy_categories": []},
    )

    assert response.status_code == 400
    assert "policy_categories" in response.get_json()["error"]
    mock_coll.insert_one.assert_not_called()
