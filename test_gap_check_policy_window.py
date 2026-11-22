"""Regression tests for gap-check document lookup and policy-window slicing.

Existing coverage pins document_id / invalid-ObjectId / positive max_policy_chars.
These cases are the remaining ways a gap check can read the wrong policy text.
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
def app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.register_blueprint(core_bp)
    return flask_app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients():
    mock_mongo = MagicMock()
    mock_anthropic = MagicMock()
    init_core(mock_mongo, MagicMock(), mock_anthropic)
    return {"mongo": mock_mongo, "anthropic": mock_anthropic}


def _collection(mock_mongo: MagicMock) -> MagicMock:
    coll = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = coll
    mock_mongo.__getitem__.return_value = db
    return coll


def test_valid_object_id_does_not_fall_through_to_string_id(app: Flask, mock_clients) -> None:
    """A 24-hex id that misses document_id and ObjectId _id is not looked up as a string _id."""
    hex_id = str(ObjectId())
    coll = _collection(mock_clients["mongo"])
    coll.find_one.side_effect = [
        None,
        None,
        {"_id": hex_id, "text": "stored under the string id"},
    ]

    with app.app_context():
        text, err = _fetch_document_text("privacy", "policies", hex_id)

    assert text is None
    assert err is not None
    response, status = err
    assert status == 404
    assert response.get_json()["error"] == "Document not found"
    assert coll.find_one.call_count == 2
    assert coll.find_one.call_args_list[0].args[0] == {"document_id": hex_id}
    assert coll.find_one.call_args_list[1].args[0] == {"_id": ObjectId(hex_id)}


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({"text": "  \n  ", "chunk_text": "from chunk"}, "from chunk"),
        ({"text": 0, "section_text": "from section"}, "from section"),
        ({"text": False, "chunk_text": "from false text"}, "from false text"),
        ({"chunk_text": 0, "section_text": "section after numeric chunk"}, "section after numeric chunk"),
    ],
)
def test_falsy_text_fields_fall_through(app: Flask, mock_clients, doc: dict, expected: str) -> None:
    """Whitespace, 0, and False do not block the next text source."""
    coll = _collection(mock_clients["mongo"])
    coll.find_one.return_value = {"document_id": "pol-1", **doc}

    with app.app_context():
        text, err = _fetch_document_text("privacy", "policies", "pol-1")

    assert err is None
    assert text == expected


def _capture_prompt(mock_anthropic: MagicMock) -> dict[str, str]:
    captured: dict[str, str] = {}

    def _create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        block = MagicMock()
        block.type = "text"
        block.text = (
            '{"addressed": false, "policy_quote": null, "missing": true, '
            '"conflict": false, "conflict_description": null}'
        )
        response = MagicMock()
        response.content = [block]
        return response

    mock_anthropic.messages.create.side_effect = _create
    return captured


def test_zero_and_negative_policy_windows_slice_by_python_rules(client, mock_clients) -> None:
    """max_policy_chars 0 keeps the whole policy; a negative value drops a prefix.

    policy[-0:] is the full string. policy[-(-n):] is policy[n:], not the last n characters.
    A JSON float is truncated with int() before that slice.
    """
    policy = "ABCDEFGHIJ"
    coll = _collection(mock_clients["mongo"])

    def _find_one(query):
        doc_id = query.get("document_id")
        if doc_id == "stat-1":
            return {"text": "Statute duty."}
        if doc_id == "pol-1":
            return {"text": policy}
        return None

    coll.find_one.side_effect = _find_one

    def _post(max_policy_chars):
        captured = _capture_prompt(mock_clients["anthropic"])
        response = client.post(
            "/gap-check",
            json={
                "policy_database_name": "pdb",
                "policy_collection_name": "policies",
                "policy_document_id": "pol-1",
                "statute_database_name": "sdb",
                "statute_collection_name": "statutes",
                "statute_document_id": "stat-1",
                "max_policy_chars": max_policy_chars,
            },
        )
        assert response.status_code == 200
        prompt = captured["prompt"]
        marker = "PRIVACY POLICY:\n"
        start = prompt.index(marker) + len(marker)
        end = prompt.index("\n\nRespond with only", start)
        return prompt[start:end]

    assert _post(0) == policy
    assert _post(-3) == policy[3:]
    assert _post(4.9) == policy[-4:]
    assert mock_clients["anthropic"].messages.create.call_count == 3
