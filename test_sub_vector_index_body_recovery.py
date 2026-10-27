"""Body recovery and document_type normalization for POST /create-sub-vector-index."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client():
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def _post(client, **kwargs):
    captured = {}

    def _start(request_dict, job_storage, flask_app):
        captured["request"] = request_dict
        captured["storage"] = job_storage
        return "job-recovered"

    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job", side_effect=_start
    ):
        response = client.post("/create-sub-vector-index", **kwargs)
    return response, captured


def test_single_quoted_body_enqueues_policy(client) -> None:
    response, captured = _post(
        client,
        data="{'document_type': 'policy'}",
        content_type="application/json",
    )
    assert response.status_code == 202
    assert response.get_json() == {"job_id": "job-recovered", "status": "pending"}
    assert captured["request"] == {
        "document_type": "policy",
        "source_query": {},
    }


def test_padded_statute_type_is_normalized(client) -> None:
    response, captured = _post(
        client,
        json={"document_type": "  STATUTE  "},
    )
    assert response.status_code == 202
    assert captured["request"]["document_type"] == "statute"
    assert captured["request"]["source_query"] == {}


def test_json_object_string_source_query_is_parsed(client) -> None:
    response, captured = _post(
        client,
        json={
            "document_type": "policy",
            "source_query": '{"document_id": "doc-9"}',
        },
    )
    assert response.status_code == 202
    assert captured["request"]["source_query"] == {"document_id": "doc-9"}


def test_null_body_skips_invalid_json_branch(client) -> None:
    """json.loads('null') succeeds, so the invalid-JSON return is skipped."""
    response, captured = _post(
        client,
        data="null",
        content_type="application/json",
    )
    assert response.status_code == 500
    assert "request" not in captured


def test_invalid_syntax_is_rejected_before_enqueue(client) -> None:
    response, captured = _post(
        client,
        data="{",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid JSON in request body"
    assert "request" not in captured


def test_blank_body_asks_for_document_type(client) -> None:
    response, captured = _post(
        client,
        data="   ",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == (
        "document_type is required and must be 'policy' or 'statute'"
    )
    assert "request" not in captured


def test_non_string_document_type_is_rejected(client) -> None:
    response, captured = _post(client, json={"document_type": True})
    assert response.status_code == 400
    assert "document_type is required" in response.get_json()["error"]
    assert "request" not in captured


def test_whitespace_document_type_is_rejected(client) -> None:
    response, captured = _post(client, json={"document_type": "   "})
    assert response.status_code == 400
    assert response.get_json()["error"] == (
        "document_type must be 'policy' or 'statute'"
    )
    assert "request" not in captured
