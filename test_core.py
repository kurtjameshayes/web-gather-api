"""Unit tests for parse-llm endpoint in core.py."""
from __future__ import annotations

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


def create_tool_use_response(sections):
    """Helper to create a mock tool use response."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "identify_sections"
    tool_use_block.input = {"sections": sections}

    mock_response = MagicMock()
    mock_response.content = [tool_use_block]
    return mock_response


class TestParseLlm:
    """Tests for the /parse-llm endpoint."""

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

        # Create mock tool response - sections start at lines 1 and 3 (with blank line between docs)
        mock_response = create_tool_use_response([
            {
                "section": "\u00a7 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "First Section",
                "start_line": 1,
            },
            {
                "section": "\u00a7 2",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "Second Section",
                "start_line": 3,
            },
        ])
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
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
        assert len(data["parsed_doc"]) == 2
        assert data["parsed_doc"][0]["section"] == "\u00a7 1"
        assert data["parsed_doc"][0]["parsed_header_text"] == "First Section"

        # Verify the collection was queried
        mock_clients["mongo"].__getitem__.assert_called_with("test_db")

        # Verify Anthropic was called with tools and line-numbered text
        call_args = mock_clients["anthropic"].messages.stream.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "First document text." in user_message
        assert "Second document text." in user_message
        assert "Third document text." in user_message
        assert "Summarize the documents" in user_message
        system_prompt = call_args[1]["system"]
        assert "legal text parser specializing in statutory interpretation" in system_prompt
        assert "Output Format" in system_prompt
        assert "line numbers at the start of each line" in system_prompt
        assert "identify_sections tool" in system_prompt
        assert "start_line (1-indexed)" in system_prompt
        # Verify tools were passed
        assert "tools" in call_args[1]
        assert call_args[1]["tools"][0]["name"] == "identify_sections"

    def test_parse_llm_missing_database(self, client, mock_clients):
        """Test error when database parameter is missing."""
        # Act
        response = client.post(
            "/parse-llm",
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
            "/parse-llm",
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
            "/parse-llm",
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
            "/parse-llm",
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
            "/parse-llm",
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

    def test_parse_llm_extracts_text_by_line_numbers(self, client, mock_clients):
        """Test that text is correctly extracted based on line numbers."""
        # Arrange - document with clear line structure
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "text": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # LLM identifies two sections: lines 1-2 and lines 3-5
        mock_response = create_tool_use_response([
            {
                "section": "\u00a7 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "First Part",
                "start_line": 1,
            },
            {
                "section": "\u00a7 2",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "Second Part",
                "start_line": 3,
            },
        ])
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Split into sections",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["parsed_doc"]) == 2
        # First section should have lines 1-2
        assert "Line 1" in data["parsed_doc"][0]["parsed_text"]
        assert "Line 2" in data["parsed_doc"][0]["parsed_text"]
        assert "Line 3" not in data["parsed_doc"][0]["parsed_text"]
        # Second section should have lines 3-5
        assert "Line 3" in data["parsed_doc"][1]["parsed_text"]
        assert "Line 4" in data["parsed_doc"][1]["parsed_text"]
        assert "Line 5" in data["parsed_doc"][1]["parsed_text"]

    def test_parse_llm_handles_llm_api_error(self, client, mock_clients):
        """Test error handling when Anthropic API fails."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"_id": "1", "text": "Test content"}]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_clients["anthropic"].messages.stream.side_effect = Exception("API Error")

        # Act
        response = client.post(
            "/parse-llm",
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

    def test_parse_llm_handles_no_tool_use(self, client, mock_clients):
        """Test error handling when LLM doesn't use the tool."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"_id": "1", "text": "Test content"}]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # Response without tool_use block
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        mock_response.content = [text_block]

        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
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

        mock_response = create_tool_use_response([
            {
                "section": "\u00a7 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "All Content",
                "start_line": 1,
            },
        ])
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
            },
        )

        # Assert
        assert response.status_code == 200

        # Verify only valid texts are in the LLM input
        call_args = mock_clients["anthropic"].messages.stream.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "Valid text" in user_message
        assert "More valid text" in user_message

    def test_parse_llm_sorts_sections_by_line_number(self, client, mock_clients):
        """Test that sections are sorted by start_line regardless of order returned."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": "1", "text": "Line 1\nLine 2\nLine 3\nLine 4"},
        ]
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # LLM returns sections out of order
        mock_response = create_tool_use_response([
            {
                "section": "\u00a7 2",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "Second",
                "start_line": 3,
            },
            {
                "section": "\u00a7 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "First",
                "start_line": 1,
            },
        ])
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Parse",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        # Sections should be sorted by line number
        assert data["parsed_doc"][0]["section"] == "\u00a7 1"
        assert data["parsed_doc"][1]["section"] == "\u00a7 2"

    def test_parse_llm_with_document_id_success(self, client, mock_clients):
        """Test successful LLM parsing with a specific document_id."""
        from bson import ObjectId

        # Arrange
        test_object_id = ObjectId()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = {
            "_id": test_object_id,
            "text": "Specific document text content.",
        }
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        mock_response = create_tool_use_response([
            {
                "section": "\u00a7 1",
                "code_name": "Civil Code",
                "jurisdiction": "California",
                "parsed_header_text": "Content",
                "start_line": 1,
            },
        ])
        mock_stream = MagicMock()
        mock_stream.get_final_message.return_value = mock_response
        mock_clients["anthropic"].messages.stream.return_value.__enter__.return_value = mock_stream

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize the document",
                "document_id": str(test_object_id),
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert "parsed_doc" in data
        assert len(data["parsed_doc"]) == 1
        assert data["parsed_doc"][0]["section"] == "\u00a7 1"

        # Verify find_one was called with the ObjectId
        mock_collection.find_one.assert_called_once()
        call_args = mock_collection.find_one.call_args[0][0]
        assert "_id" in call_args
        assert call_args["_id"] == test_object_id

        # Verify the specific document text is in the LLM input
        call_args = mock_clients["anthropic"].messages.stream.call_args
        user_message = call_args[1]["messages"][0]["content"]
        assert "Specific document text content." in user_message

    def test_parse_llm_with_document_id_not_found(self, client, mock_clients):
        """Test error when document_id is not found in the collection."""
        from bson import ObjectId

        # Arrange
        test_object_id = ObjectId()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
                "document_id": str(test_object_id),
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_parse_llm_with_invalid_document_id_format(self, client, mock_clients):
        """Test error when document_id has invalid ObjectId format."""
        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
                "document_id": "invalid-object-id",
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "invalid" in data["error"].lower()

    def test_parse_llm_with_document_id_no_text(self, client, mock_clients):
        """Test error when document found by document_id has no text attribute."""
        from bson import ObjectId

        # Arrange
        test_object_id = ObjectId()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = {
            "_id": test_object_id,
            "title": "Document without text",
        }
        mock_clients["mongo"].__getitem__.return_value.__getitem__.return_value = mock_collection

        # Act
        response = client.post(
            "/parse-llm",
            json={
                "database": "test_db",
                "collection": "test_collection",
                "parse_prompt": "Summarize",
                "document_id": str(test_object_id),
            },
        )

        # Assert
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "no text" in data["error"].lower()
