"""Regression tests for core._fetch_document_text document lookup and text extraction."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

# Avoid loading sentence_transformers / pyppeteer in unit tests.
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

from core import _fetch_document_text, init_core


@pytest.fixture
def mock_mongo() -> MagicMock:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return mock_mongo


def _wire_collection(mock_mongo: MagicMock, coll: MagicMock) -> None:
    db = MagicMock()
    db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = db


def test_fetch_document_text_by_document_id_field(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.return_value = {"document_id": "pol-1", "text": "  Full policy text.  "}
    _wire_collection(mock_mongo, coll)

    text, err = _fetch_document_text("privacy", "policies", "pol-1")

    assert err is None
    assert text == "Full policy text."
    coll.find_one.assert_called_once_with({"document_id": "pol-1"})


def test_fetch_document_text_falls_back_to_object_id(mock_mongo: MagicMock) -> None:
    oid = ObjectId()
    coll = MagicMock()
    coll.find_one.side_effect = [None, {"_id": oid, "text": "By ObjectId"}]
    _wire_collection(mock_mongo, coll)

    text, err = _fetch_document_text("privacy", "policies", str(oid))

    assert err is None
    assert text == "By ObjectId"
    assert coll.find_one.call_count == 2
    assert coll.find_one.call_args_list[1].args[0] == {"_id": oid}


def test_fetch_document_text_falls_back_to_string_id(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.side_effect = [None, {"_id": "legacy-id", "chunk_text": "Chunk body"}]
    _wire_collection(mock_mongo, coll)

    text, err = _fetch_document_text("privacy", "policies", "legacy-id")

    assert err is None
    assert text == "Chunk body"
    # Invalid ObjectId path uses string _id lookup
    assert coll.find_one.call_args_list[1].args[0] == {"_id": "legacy-id"}


def test_fetch_document_text_prefers_section_text_over_policy_chunks(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.return_value = {
        "document_id": "pol-2",
        "section_text": "Section body",
        "policy_chunks": [{"chunk_text": "Should not win"}],
    }
    _wire_collection(mock_mongo, coll)

    text, err = _fetch_document_text("privacy", "policies", "pol-2")

    assert err is None
    assert text == "Section body"


def test_fetch_document_text_assembles_policy_chunks(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.return_value = {
        "document_id": "pol-3",
        "policy_chunks": [
            {"chunk_text": "First chunk"},
            {"chunk_header_text": "Header only"},
            {"chunk_text": "  "},
            "skip-non-dict",
            {"chunk_text": "Third chunk"},
        ],
    }
    _wire_collection(mock_mongo, coll)

    text, err = _fetch_document_text("privacy", "policies", "pol-3")

    assert err is None
    assert text == "First chunk\n\nHeader only\n\nThird chunk"


def test_fetch_document_text_not_found(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.return_value = None
    _wire_collection(mock_mongo, coll)

    app = Flask(__name__)
    with app.app_context():
        text, err = _fetch_document_text("privacy", "policies", "missing")

    assert text is None
    assert err is not None
    response, status = err
    assert status == 404
    assert response.get_json()["error"] == "Document not found"


def test_fetch_document_text_no_extractable_text(mock_mongo: MagicMock) -> None:
    coll = MagicMock()
    coll.find_one.return_value = {"document_id": "empty", "policy_chunks": []}
    _wire_collection(mock_mongo, coll)

    app = Flask(__name__)
    with app.app_context():
        text, err = _fetch_document_text("privacy", "policies", "empty")

    assert text is None
    assert err is not None
    response, status = err
    assert status == 404
    assert "no extractable text" in response.get_json()["error"].lower()
