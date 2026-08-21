"""Regression tests for /create-sub-vector-index JSON5 parsing and init errors."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())

from core import core_bp, init_core


@pytest.fixture
def client() -> Any:
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app.test_client()


def test_create_sub_vector_index_accepts_json5_trailing_comma(client) -> None:
    """JSON5 trailing-comma bodies still start a job after standard json.loads fails."""
    mock_storage = MagicMock()
    with patch("core._get_index_job_storage", return_value=mock_storage), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-json5-1"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            data='{ "document_type": "Statute", "source_query": {"document_id": "doc-1"}, }',
            content_type="application/json",
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-json5-1"
    start_job.assert_called_once_with(
        request_dict={
            "document_type": "statute",
            "source_query": {"document_id": "doc-1"},
        },
        job_storage=mock_storage,
        flask_app=ANY,
    )


def test_create_sub_vector_index_uninitialized_storage_returns_500(client) -> None:
    """RuntimeError from uninitialized job storage maps to HTTP 500."""
    with patch(
        "core._get_index_job_storage",
        side_effect=RuntimeError(
            "Index job service not initialized; call init_index_job first"
        ),
    ):
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy"},
        )

    assert response.status_code == 500
    assert "not initialized" in response.get_json()["error"].lower()
