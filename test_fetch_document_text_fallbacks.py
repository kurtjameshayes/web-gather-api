"""_fetch_document_text lookup and field-fallback contracts used by /gap-check.

Gap-check remaps this helper's empty-text 404 to a generic 'document not found'
(#221). These tests pin the helper itself so a fallback-order change would send
the wrong statute/policy text to the LLM.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import _fetch_document_text, core_bp, init_core


@pytest.fixture
def mock_coll():
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = coll
    return coll


@pytest.fixture
def app_ctx():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    with app.app_context():
        yield app


def test_fetch_document_text_prefers_text_over_chunk_text(mock_coll, app_ctx) -> None:
    mock_coll.find_one.return_value = {
        "document_id": "doc-1",
        "text": "  primary text  ",
        "chunk_text": "chunk fallback",
        "section_text": "section fallback",
    }

    text, err = _fetch_document_text("db", "coll", "doc-1")

    assert err is None
    assert text == "primary text"
    mock_coll.find_one.assert_called_once_with({"document_id": "doc-1"})


def test_fetch_document_text_prefers_chunk_text_over_section_text(mock_coll, app_ctx) -> None:
    mock_coll.find_one.return_value = {
        "document_id": "doc-1",
        "text": "   ",
        "chunk_text": "from chunk_text",
        "section_text": "from section_text",
    }

    text, err = _fetch_document_text("db", "coll", "doc-1")

    assert err is None
    assert text == "from chunk_text"


def test_fetch_document_text_uses_section_text_when_chunk_text_blank(mock_coll, app_ctx) -> None:
    mock_coll.find_one.return_value = {
        "document_id": "doc-1",
        "chunk_text": "",
        "section_text": "  section only  ",
    }

    text, err = _fetch_document_text("db", "coll", "doc-1")

    assert err is None
    assert text == "section only"


def test_fetch_document_text_assembles_policy_chunks(mock_coll, app_ctx) -> None:
    mock_coll.find_one.return_value = {
        "document_id": "doc-1",
        "policy_chunks": [
            "skip-string",
            {"chunk_header_text": "header only"},
            {"chunk_text": "  body  ", "chunk_header_text": "ignored header"},
            {"chunk_text": "   "},
            None,
        ],
    }

    text, err = _fetch_document_text("db", "coll", "doc-1")

    assert err is None
    assert text == "header only\n\nbody"


def test_fetch_document_text_falls_back_to_objectid_then_string_id(mock_coll, app_ctx) -> None:
    oid = ObjectId()
    mock_coll.find_one.side_effect = [
        None,
        {"_id": oid, "text": "via object id"},
    ]

    text, err = _fetch_document_text("db", "coll", str(oid))

    assert err is None
    assert text == "via object id"
    assert mock_coll.find_one.call_args_list[0][0][0] == {"document_id": str(oid)}
    assert mock_coll.find_one.call_args_list[1][0][0] == {"_id": oid}


def test_fetch_document_text_string_id_fallback_when_oid_invalid(mock_coll, app_ctx) -> None:
    mock_coll.find_one.side_effect = [
        None,
        {"_id": "ingest-uuid", "text": "via string id"},
    ]

    text, err = _fetch_document_text("db", "coll", "ingest-uuid")

    assert err is None
    assert text == "via string id"
    assert mock_coll.find_one.call_args_list[1][0][0] == {"_id": "ingest-uuid"}


def test_fetch_document_text_empty_extractable_is_helper_404(mock_coll, app_ctx) -> None:
    """Helper reports 'no extractable text'; gap-check remaps this later."""
    mock_coll.find_one.return_value = {
        "document_id": "doc-1",
        "title": "no body",
        "policy_chunks": [{"other": "x"}],
    }

    text, err = _fetch_document_text("db", "coll", "doc-1")

    assert text is None
    assert err is not None
    resp, code = err
    assert code == 404
    assert "no extractable text" in resp.get_json()["error"].lower()
