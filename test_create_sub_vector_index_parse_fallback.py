"""Regression tests for /create-sub-vector-index body parsing leftovers.

JSON5 trailing-comma bodies are covered on other coverage branches. These tests
pin the Swagger single-quote regex fallback (when json5 is unavailable) and the
invalid source_query-as-string 400.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client() -> Any:
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


@pytest.fixture
def disable_json5():
    """Force the regex fallback by making json5.loads fail."""
    previous = sys.modules.get("json5")
    fake = MagicMock()
    fake.loads.side_effect = ValueError("json5 unavailable")
    sys.modules["json5"] = fake
    yield
    if previous is None:
        sys.modules.pop("json5", None)
    else:
        sys.modules["json5"] = previous


def test_create_sub_vector_index_accepts_single_quoted_swagger_body(
    client, disable_json5
) -> None:
    """Copied Swagger bodies with single-quoted keys must still start a job."""
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-sq-1"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            data="{'document_type': 'policy'}",
            content_type="application/json",
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-sq-1"
    start_job.assert_called_once_with(
        request_dict={"document_type": "policy", "source_query": {}},
        job_storage=mock_storage,
        flask_app=ANY,
    )


def test_create_sub_vector_index_unparseable_body_is_400(client, disable_json5) -> None:
    """Bodies that are neither JSON, JSON5, nor single-quoted key/value pairs are 400."""
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            data="document_type=policy",
            content_type="application/json",
        )

    assert response.status_code == 400
    assert "Invalid JSON" in response.get_json()["error"]
    start_job.assert_not_called()


def test_create_sub_vector_index_invalid_source_query_string_is_400(client) -> None:
    """A source_query JSON string that does not parse is 400, not coerced to {}."""
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": "{not-json"},
        )

    assert response.status_code == 400
    assert "source_query" in response.get_json()["error"]
    start_job.assert_not_called()
