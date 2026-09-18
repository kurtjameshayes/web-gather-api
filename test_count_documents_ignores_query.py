"""Pin GET /count-documents ignoring the query parameter.

Unlike GET /documents, count-documents always calls count_documents({}) and does
not parse, reject, or apply `query`. A filter (including $where) is currently a
no-op, so the count is collection-wide.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from db import db_bp, init_db


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_mongo_client() -> MagicMock:
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


def test_count_documents_ignores_query_including_where(client, mock_mongo_client) -> None:
    mock_collection = MagicMock()
    mock_collection.count_documents.return_value = 7
    mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection

    response = client.get(
        '/count-documents?database_name=policies&collection_name=docs'
        '&query={"$where":"1==1","status":"active"}'
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 7
    mock_collection.count_documents.assert_called_once_with({})
