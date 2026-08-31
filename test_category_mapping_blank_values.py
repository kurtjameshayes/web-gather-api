"""Regression tests for POST /category-mapping blank-after-strip persistence.

Whitespace-only statute_category is truthy so it passes the required-field
check, then strip() stores an empty string. Empty-string optional fields are
not None, so they are written too. Pin that contract; do not add a non-empty
guard in a coverage-only change.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def mock_coll() -> MagicMock:
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    pc_db = MagicMock()
    coll = MagicMock()
    pc_db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = pc_db
    coll.insert_one.return_value.inserted_id = ObjectId()
    return coll


@pytest.fixture
def client(mock_coll: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_post_category_mapping_whitespace_statute_category_inserts_empty(
    client, mock_coll
) -> None:
    """Whitespace-only statute_category currently inserts statute_category=''."""
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "   ",
            "policy_categories": ["notice"],
        },
    )

    assert response.status_code == 201
    inserted = mock_coll.insert_one.call_args[0][0]
    assert inserted["statute_category"] == ""
    assert inserted["policy_categories"] == ["notice"]


def test_post_category_mapping_blank_policy_category_items_are_kept(
    client, mock_coll
) -> None:
    """Whitespace-only policy_categories items are stripped to empty strings, not dropped."""
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "consent",
            "policy_categories": ["  ", "opt_in", ""],
        },
    )

    assert response.status_code == 201
    inserted = mock_coll.insert_one.call_args[0][0]
    assert inserted["policy_categories"] == ["", "opt_in", ""]


def test_post_category_mapping_empty_string_optional_fields_are_written(
    client, mock_coll
) -> None:
    """Empty-string sub_topic/description are persisted; JSON null is omitted."""
    response = client.post(
        "/category-mapping",
        json={
            "statute_category": "retention",
            "policy_categories": ["keep"],
            "sub_topic": "",
            "description": None,
        },
    )

    assert response.status_code == 201
    inserted = mock_coll.insert_one.call_args[0][0]
    assert inserted["sub_topic"] == ""
    assert "description" not in inserted
