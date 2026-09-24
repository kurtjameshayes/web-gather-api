"""Regression tests for shared text-splitting strategies.

`chunk_text` is the splitter behind POST /create-chunks (always character) and
`index_document_chunks` (caller-selected strategy). These tests pin overlap,
fallback, and semantic-break behavior that changes what text is stored.
"""
from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core
from core import core_bp, init_core


def test_character_chunks_share_exact_overlap_suffix() -> None:
    """Default character windows repeat the overlap suffix as the next prefix."""
    chunks = core.chunk_text_by_character("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    assert chunks[0][-3:] == chunks[1][:3]
    assert chunks[1][-3:] == chunks[2][:3]
    assert chunks[2][-3:] == chunks[3][:3]


def test_character_max_safe_overlap_still_covers_every_character() -> None:
    """overlap == chunk_size - 1 advances one character and does not drop text."""
    text = "abcdef"
    chunks = core.chunk_text_by_character(text, chunk_size=3, overlap=2)

    assert chunks == ["abc", "bcd", "cde", "def"]
    assert "".join(dict.fromkeys("".join(chunks))) == text


def test_character_zero_overlap_partitions_without_gaps() -> None:
    chunks = core.chunk_text_by_character("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=0)

    assert chunks == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert "".join(chunks) == "abcdefghijklmnopqrstuvwxyz"


def test_character_negative_overlap_skips_the_gap_and_tail() -> None:
    """Negative overlap is accepted and jumps the window forward, dropping text.

    chunk_size=10, overlap=-3 on a 26-char string stores [0:10] then [13:23]
    and never returns the skipped middle or the tail past the jump.
    """
    chunks = core.chunk_text_by_character("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=-3)

    assert chunks == ["abcdefghij", "nopqrstuvw"]
    assert "klm" not in "".join(chunks)
    assert "xyz" not in "".join(chunks)


def test_character_whitespace_only_and_empty_return_no_chunks() -> None:
    assert core.chunk_text_by_character("", chunk_size=10, overlap=0) == []
    assert core.chunk_text_by_character(None, chunk_size=10, overlap=0) == []
    assert core.chunk_text_by_character("   \n  ", chunk_size=4, overlap=0) == []


def test_character_windows_are_stripped() -> None:
    """Spaces on a window edge are removed, so they are not stored in the chunk."""
    assert core.chunk_text_by_character("abcd    efgh", chunk_size=6, overlap=0) == ["abcd", "efgh"]


def test_sentence_overlap_keeps_trailing_sentences_that_fit() -> None:
    """A sentence is carried forward only when its length fits in overlap."""
    chunks = core.chunk_text_by_sentence("AAAA. BBBB. CCCC. DDDD.", chunk_size=10, overlap=5)

    assert chunks == ["AAAA.", "AAAA. BBBB.", "BBBB. CCCC.", "CCCC. DDDD."]


def test_sentence_negative_overlap_does_not_drop_sentences() -> None:
    """Unlike character mode, a negative overlap only disables sentence carry-forward."""
    chunks = core.chunk_text_by_sentence("AAAA. BBBB. CCCC. DDDD.", chunk_size=10, overlap=-5)

    assert chunks == ["AAAA.", "BBBB.", "CCCC.", "DDDD."]


def test_sentence_strategy_keeps_an_oversized_sentence_intact() -> None:
    long_sentence = "This sentence is definitely longer than twenty characters."
    chunks = core.chunk_text_by_sentence(f"{long_sentence} Short.", chunk_size=20, overlap=0)

    assert chunks == [long_sentence, "Short."]


def test_unknown_strategy_name_falls_back_to_character() -> None:
    """Strategy keys are case-sensitive; 'Sentence' is not the sentence splitter."""
    text = "abcdefghij"
    expected = core.chunk_text_by_character(text, chunk_size=4, overlap=1)

    assert core.chunk_text(text, chunk_size=4, overlap=1, strategy="Sentence") == expected
    assert core.chunk_text(text, chunk_size=4, overlap=1, strategy="nope") == expected
    assert expected == ["abcd", "defg", "ghij"]


def test_semantic_single_sentence_does_not_encode() -> None:
    model = MagicMock()

    chunks = core.chunk_text_by_semantic("Only one sentence here.", chunk_size=50, overlap=0, model=model)

    assert chunks == ["Only one sentence here."]
    model.encode.assert_not_called()


def _similarity_model() -> MagicMock:
    """Three unit vectors with adjacent dots ~0.95 then ~0.05."""
    first = np.array([1.0, 0.0])
    second = np.array([0.95, math.sqrt(1 - 0.95**2)])
    perpendicular = np.array([-second[1], second[0]])
    third = 0.05 * second + math.sqrt(1 - 0.05**2) * perpendicular
    model = MagicMock()
    model.encode.return_value = np.vstack([first, second, third])
    return model


def test_semantic_break_requires_low_similarity_and_minimum_chunk_size() -> None:
    """A similarity drop splits only after the chunk exceeds 20% of chunk_size."""
    text = (
        "Alpha discusses consent requirements. "
        "Beta discusses consent requirements. "
        "Gamma discusses unrelated retention."
    )
    model = _similarity_model()

    split = core.chunk_text_by_semantic(text, chunk_size=200, overlap=0, model=model)

    assert split == [
        "Alpha discusses consent requirements. Beta discusses consent requirements.",
        "Gamma discusses unrelated retention.",
    ]
    model.encode.assert_called_once()
    assert model.encode.call_args.kwargs["convert_to_numpy"] is True
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True

    # Same vectors, but the current chunk is still under 20% of chunk_size.
    kept = core.chunk_text_by_semantic(text, chunk_size=1000, overlap=0, model=_similarity_model())
    assert kept == [
        "Alpha discusses consent requirements. "
        "Beta discusses consent requirements. "
        "Gamma discusses unrelated retention."
    ]


def test_semantic_without_model_falls_back_to_sentence() -> None:
    text = "Alpha discusses consent requirements. Beta discusses consent requirements."
    assert core.chunk_text(text, chunk_size=40, overlap=0, strategy="semantic") == core.chunk_text_by_sentence(
        text, chunk_size=40, overlap=0
    )


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_create_chunks_negative_overlap_omits_skipped_spans(client) -> None:
    """POST /create-chunks accepts a negative overlap and stores only the windows that remain."""
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    source_coll = MagicMock()
    dest_coll = MagicMock()
    db = MagicMock()

    def _get_collection(name: str) -> MagicMock:
        if name == "docs":
            return source_coll
        if name == "chunks":
            return dest_coll
        return MagicMock()

    db.__getitem__.side_effect = _get_collection
    mock_mongo.__getitem__.return_value = db
    source_coll.find.return_value = [
        {"_id": "doc-1", "text": "abcdefghijklmnopqrstuvwxyz", "document_id": "policy-1"}
    ]

    response = client.post(
        "/create-chunks",
        json={
            "database": "privacy-db",
            "source_collection": "docs",
            "destination_collection": "chunks",
            "source_column": "text",
            "chunk_column": "chunk_text",
            "chunk_size": 10,
            "overlap": -3,
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["overlap"] == -3
    assert body["chunk_size"] == 10
    assert body["records_inserted"] == 2
    assert body["source_rows_processed"] == 1
    assert body["source_rows_skipped"] == 0
    stored = [call.args[0]["chunk_text"] for call in dest_coll.insert_one.call_args_list]
    assert stored == ["abcdefghij", "nopqrstuvw"]
    assert all(call.args[0]["source_id"] == "doc-1" for call in dest_coll.insert_one.call_args_list)
    assert [call.args[0]["chunk_index"] for call in dest_coll.insert_one.call_args_list] == [0, 1]
