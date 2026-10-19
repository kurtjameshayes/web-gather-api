"""GET /documents clamps page size and floors offset before querying Mongo."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app.test_client()


@pytest.fixture
def mock_mongo_client() -> MagicMock:
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


def _wire_collection(mock_mongo: MagicMock) -> MagicMock:
    collection = MagicMock()
    cursor = MagicMock()
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    collection.find.return_value = cursor
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection
    return collection


@pytest.mark.parametrize(
    ("page_query", "expected_limit", "expected_offset"),
    [
        ("", 100, 0),
        ("limit=0&offset=-3", 1, 0),
        ("limit=-4&offset=-1", 1, 0),
        ("limit=100000&offset=40", 10000, 40),
        ("limit=10000&offset=0", 10000, 0),
        ("limit=abc&offset=1.5", 100, 0),
        ("limit=%2025%20&offset=%208%20", 25, 8),
        ("limit=&offset=", 100, 0),
        ("limit=1&offset=2", 1, 2),
    ],
)
def test_list_documents_clamps_limit_and_floors_offset(
    client,
    mock_mongo_client: MagicMock,
    page_query: str,
    expected_limit: int,
    expected_offset: int,
) -> None:
    """Garbage or extreme page params still query, with limit in 1..10000 and offset >= 0.

    Offset is not capped. A non-numeric limit or offset falls back to 100 and 0.
    """
    collection = _wire_collection(mock_mongo_client)
    url = "/documents?database_name=test_db&collection_name=test_collection"
    if page_query:
        url = f"{url}&{page_query}"

    response = client.get(url)

    assert response.status_code == 200
    payload: Dict[str, Any] = response.get_json()
    assert payload["documents"] == []
    assert payload["limit"] == expected_limit
    assert payload["offset"] == expected_offset
    collection.find.assert_called_once_with({})
    cursor = collection.find.return_value
    cursor.skip.assert_called_once_with(expected_offset)
    cursor.limit.assert_called_once_with(expected_limit)
