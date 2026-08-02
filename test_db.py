"""Unit tests for database endpoints in db.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from bson import ObjectId
from flask import Flask

from db import _serialize_doc, db_bp, init_db


def test_serialize_doc_stringifies_object_id() -> None:
    """API responses must stringify Mongo _id values for JSON safety."""
    oid = ObjectId()
    doc = {"_id": oid, "name": "policy"}
    _serialize_doc(doc)
    assert doc["_id"] == str(oid)
    assert doc["name"] == "policy"


def test_serialize_doc_ignores_missing_id() -> None:
    doc = {"name": "policy"}
    _serialize_doc(doc)
    assert doc == {"name": "policy"}


@pytest.fixture
def app():
    """Create a test Flask application."""
    app = Flask(__name__)
    app.register_blueprint(db_bp)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client."""
    mock_client = MagicMock()
    init_db(mock_client)
    return mock_client


class TestCountDocuments:
    """Tests for the /count-documents endpoint."""

    def test_count_documents_success(self, client, mock_mongo_client):
        """Test successful document count retrieval."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 42
        mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.get(
            "/count-documents?database_name=test_db&collection_name=test_collection"
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["database_name"] == "test_db"
        assert data["collection_name"] == "test_collection"
        assert data["count"] == 42

    def test_count_documents_missing_database_name(self, client, mock_mongo_client):
        """Test error when database_name is missing."""
        # Act
        response = client.get("/count-documents?collection_name=test_collection")

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "database_name" in data["error"]

    def test_count_documents_missing_collection_name(self, client, mock_mongo_client):
        """Test error when collection_name is missing."""
        # Act
        response = client.get("/count-documents?database_name=test_db")

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "collection_name" in data["error"]

    def test_count_documents_missing_both_params(self, client, mock_mongo_client):
        """Test error when both parameters are missing."""
        # Act
        response = client.get("/count-documents")

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_count_documents_empty_collection(self, client, mock_mongo_client):
        """Test counting documents in an empty collection."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 0
        mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.get(
            "/count-documents?database_name=test_db&collection_name=empty_collection"
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["count"] == 0

    def test_count_documents_large_count(self, client, mock_mongo_client):
        """Test counting a large number of documents."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 1_000_000
        mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.get(
            "/count-documents?database_name=test_db&collection_name=large_collection"
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["count"] == 1_000_000

    def test_count_documents_special_characters_in_names(self, client, mock_mongo_client):
        """Test with special characters in database and collection names."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 5
        mock_mongo_client.__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.get(
            "/count-documents?database_name=my-db_123&collection_name=my-collection_456"
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["database_name"] == "my-db_123"
        assert data["collection_name"] == "my-collection_456"
        assert data["count"] == 5
