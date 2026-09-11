"""POST /category-mapping treats an empty policy_categories list as missing."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def client():
    init_db(MagicMock())
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_post_category_mapping_empty_list_is_required_error(client) -> None:
    """[] is falsy, so the handler returns 'policy_categories is required' rather than the array-type error."""
    response = client.post(
        "/category-mapping",
        json={"statute_category": "consumer_rights", "policy_categories": []},
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "required" in error
    assert "array" not in error


def test_post_category_mapping_non_list_still_uses_array_error(client) -> None:
    """A truthy non-list still hits the array-type guard, distinct from the empty-list required path."""
    response = client.post(
        "/category-mapping",
        json={"statute_category": "consumer_rights", "policy_categories": "not-a-list"},
    )
    assert response.status_code == 400
    assert "array" in response.get_json()["error"]
