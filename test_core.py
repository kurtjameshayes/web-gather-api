"""Unit tests for parse_llm endpoint in core.py."""
import json
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock sentence_transformers before importing core
sys.modules['sentence_transformers'] = MagicMock()

from flask import Flask

from core import core_bp, init_core


@pytest.fixture
def app():
    """Create a test Flask application."""
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def mock_clients():
    """Create mock MongoDB, Firecrawl, and Anthropic clients."""
    mock_mongo = MagicMock()
    mock_firecrawl = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, mock_firecrawl, mock_anthropic)
    return {
        "mongo": mock_mongo,
        "firecrawl": mock_firecrawl,
        "anthropic": mock_anthropic,
    }


class TestParseLlm:
    """Tests for the /parse_llm endpoint."""

    def test_parse_llm_success(self, client, mock_clients):
        """Test successful LLM parsing with database and collection."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "text": "First document text."},
            {"_id": "2", "text": "Second document text."},
            {"_id": "3", "text": "Third document text."},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"parsed_doc": [{"document_id": "doc_001", "parsed_header_text": "Summary", "parsed_text": "Combined content"}]}')]
        mock_clients["anthropic"].messages.create.return_value = mock_response

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize the documents",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert "parsed_doc" in data
        assert len(data["parsed_doc"]) == 1

        # Verify the collection was queried
        mock_clients["mongo"].__getitem__.assert_called_with("test_db")

        # Verify Anthropic was called with concatenated text
        call_args = mock_clients["anthropic"].messages.create.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "First document text." in user_message
        assert "Second document text." in user_message
        assert "Third document text." in user_message
        assert "Summarize the documents" in user_message

    def test_parse_llm_missing_database(self, client, mock_clients):
        """Test error when database parameter is missing."""
        # Act
        response = client.post(
            "/parse_llm",
            json={
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "database" in data["error"]

    def test_parse_llm_missing_collection(self, client, mock_clients):
        """Test error when collection parameter is missing."""
        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "collection" in data["error"]

    def test_parse_llm_missing_parse_prompt(self, client, mock_clients):
        """Test error when parse_prompt parameter is missing."""
        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "parse_prompt" in data["error"]

    def test_parse_llm_empty_collection(self, client, mock_clients):
        """Test error when collection has no documents."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "empty_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "no documents" in data["error"].lower() or "no text" in data["error"].lower()

    def test_parse_llm_documents_without_text(self, client, mock_clients):
        """Test error when documents have no text attribute."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "title": "Doc without text"},
            {"_id": "2", "content": "Wrong attribute name"},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_parse_llm_concatenates_text_correctly(self, client, mock_clients):
        """Test that multiple document texts are concatenated correctly."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "text": "Alpha"},
            {"_id": "2", "text": "Beta"},
            {"_id": "3", "text": "Gamma"},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"parsed_doc": []}')]
        mock_clients["anthropic"].messages.create.return_value = mock_response

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Parse this",
            },
        )

        # Assert
        assert response.status_code == 200

        # Verify all texts are in the LLM input
        call_args = mock_clients["anthropic"].messages.create.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "Alpha" in user_message
        assert "Beta" in user_message
        assert "Gamma" in user_message

    def test_parse_llm_handles_llm_json_error(self, client, mock_clients):
        """Test error handling when LLM returns invalid JSON."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"_id": "1", "text": "Test content"}]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]
        mock_clients["anthropic"].messages.create.return_value = mock_response

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data

    def test_parse_llm_handles_llm_api_error(self, client, mock_clients):
        """Test error handling when Anthropic API fails."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"_id": "1", "text": "Test content"}]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_clients["anthropic"].messages.create.side_effect = Exception("API Error")

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data

    def test_parse_llm_skips_documents_with_empty_text(self, client, mock_clients):
        """Test that documents with empty or whitespace-only text are skipped."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "text": "Valid text"},
            {"_id": "2", "text": ""},
            {"_id": "3", "text": "   "},
            {"_id": "4", "text": "More valid text"},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"parsed_doc": []}')]
        mock_clients["anthropic"].messages.create.return_value = mock_response

        # Act
        response = client.post(
            "/parse_llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 200

        # Verify only valid texts are in the LLM input
        call_args = mock_clients["anthropic"].messages.create.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "Valid text" in user_message
        assert "More valid text" in user_message
