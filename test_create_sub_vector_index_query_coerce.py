"""Leftover create-sub-vector-index source_query coercion into unscoped {} jobs."""
from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


@pytest.fixture
def mock_clients() -> Dict[str, Any]:
    mock_mongo = MagicMock()
    init_core(mock_mongo, MagicMock(), MagicMock())
    return {"mongo": mock_mongo}


def _start_job(**kwargs: Any) -> str:
    return "job-coerced"


def test_list_source_query_is_coerced_to_empty_and_enqueued(
    client, mock_clients
) -> None:
    """A JSON-array source_query is not 400; it becomes {} and the job is still 202."""
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", side_effect=_start_job
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": [1, 2]},
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-coerced"
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["document_type"] == "policy"
    assert request_dict["source_query"] == {}


def test_json_string_list_source_query_is_coerced_after_parse(
    client, mock_clients
) -> None:
    """A JSON-string list parses, then the non-dict result is coerced to {}."""
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-str-list"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "statute", "source_query": "[1, 2]"},
        )

    assert response.status_code == 202
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["document_type"] == "statute"
    assert request_dict["source_query"] == {}


def test_whitespace_source_query_string_is_coerced_to_empty(
    client, mock_clients
) -> None:
    """Whitespace strings skip json.loads (strip is falsy) then coerce to {}."""
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-ws"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": "   "},
        )

    assert response.status_code == 202
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["source_query"] == {}
