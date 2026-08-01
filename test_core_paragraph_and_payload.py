"""Regression tests for paragraph chunking and request JSON payload helpers."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from flask import Flask

# Avoid loading sentence_transformers in unit tests.
sys.modules["sentence_transformers"] = MagicMock()

import core
from core import core_bp


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


def test_split_into_paragraph_chunks_basic() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    assert core._split_into_paragraph_chunks(text) == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_split_into_paragraph_chunks_empty_and_whitespace() -> None:
    assert core._split_into_paragraph_chunks("") == []
    assert core._split_into_paragraph_chunks("   \n\n  \n") == []


def test_chunk_text_by_paragraph_empty() -> None:
    assert core.chunk_text_by_paragraph("") == []
    assert core.chunk_text_by_paragraph("\n\n") == []


def test_chunk_text_by_paragraph_single_chunk() -> None:
    text = "Alpha paragraph.\n\nBeta paragraph."
    chunks = core.chunk_text_by_paragraph(text, chunk_size=200, overlap=20)
    assert chunks == [text]


def test_chunk_text_by_paragraph_splits_and_overlaps() -> None:
    p1 = "A" * 40
    p2 = "B" * 40
    p3 = "C" * 40
    text = f"{p1}\n\n{p2}\n\n{p3}"
    chunks = core.chunk_text_by_paragraph(text, chunk_size=90, overlap=50)
    assert len(chunks) >= 2
    assert p1 in chunks[0]
    assert p2 in chunks[0]
    # Overlap should retain a trailing paragraph from the prior chunk.
    assert p2 in chunks[1]
    assert p3 in chunks[1]


def test_get_json_payload_valid_dict(app: Flask) -> None:
    with app.test_request_context(
        "/create-chunks",
        method="POST",
        data='{"database":"db"}',
        content_type="application/json",
    ):
        payload, err = core._get_json_payload_or_error()
    assert err is None
    assert payload == {"database": "db"}


def test_get_json_payload_empty_body(app: Flask) -> None:
    with app.test_request_context("/create-chunks", method="POST", data=""):
        payload, err = core._get_json_payload_or_error()
    assert err is None
    assert payload == {}


def test_get_json_payload_invalid_json_returns_400(app: Flask) -> None:
    with app.test_request_context(
        "/create-chunks",
        method="POST",
        data="{bad json",
        content_type="application/json",
    ):
        payload, err = core._get_json_payload_or_error()
    assert payload is None
    assert err is not None
    response, status = err
    assert status == 400
    assert "Invalid JSON" in response.get_json()["error"]


def test_get_json_payload_non_dict_json_becomes_empty_dict(app: Flask) -> None:
    with app.test_request_context(
        "/create-chunks",
        method="POST",
        data='["not", "an", "object"]',
        content_type="application/json",
    ):
        payload, err = core._get_json_payload_or_error()
    assert err is None
    assert payload == {}
