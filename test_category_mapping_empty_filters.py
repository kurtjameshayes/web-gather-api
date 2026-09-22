"""GET /category-mapping empty statute_category=/sub_topic= are unfiltered.

Truthy filter params are applied as equality. Empty strings are falsy, so
`statute_category=` / `sub_topic=` currently run find({}) rather than matching
documents whose category/topic is the empty string. Distinct from POST blank
strip-to-empty persistence (#221) and query-param override (#217).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import PRIVACY_COMPLIANCE_DB, db_bp, init_db


@pytest.fixture
def mock_coll() -> MagicMock:
    mock_mongo = MagicMock()
    init_db(mock_mongo)
    pc_db = MagicMock()
    coll = MagicMock()

    def _get_db(name: str) -> MagicMock:
        return pc_db if name == PRIVACY_COMPLIANCE_DB else MagicMock()

    mock_mongo.__getitem__.side_effect = _get_db
    pc_db.__getitem__.return_value = coll
    coll.find.return_value = [{"_id": "m1", "statute_category": "consent"}]
    return coll


@pytest.fixture
def client(mock_coll: MagicMock):
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


def test_get_category_mapping_empty_statute_category_is_unfiltered(client, mock_coll) -> None:
    """Empty statute_category= does not add an equality filter (find({}))."""
    response = client.get("/category-mapping?statute_category=")

    assert response.status_code == 200
    mock_coll.find.assert_called_once_with({})


def test_get_category_mapping_empty_sub_topic_is_unfiltered(client, mock_coll) -> None:
    """Empty sub_topic= is falsy and does not filter; non-empty values still do."""
    empty = client.get("/category-mapping?sub_topic=")
    assert empty.status_code == 200
    mock_coll.find.assert_called_with({})

    mock_coll.find.reset_mock()
    mock_coll.find.return_value = []
    filtered = client.get("/category-mapping?sub_topic=notice")
    assert filtered.status_code == 200
    mock_coll.find.assert_called_once_with({"sub_topic": "notice"})


def test_get_category_mapping_whitespace_filters_are_forwarded(client, mock_coll) -> None:
    """Whitespace-only filters are truthy and forwarded as exact equality."""
    response = client.get("/category-mapping?statute_category=%20%20&sub_topic=%20")

    assert response.status_code == 200
    mock_coll.find.assert_called_once_with(
        {"statute_category": "  ", "sub_topic": " "}
    )
