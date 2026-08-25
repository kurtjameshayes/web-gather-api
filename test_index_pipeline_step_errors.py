"""Regression tests for index_job_service._run_pipeline_step error mapping.

Sub-vector-index jobs fail closed when a pipeline endpoint returns non-2xx.
The raised RuntimeError must include the step name, path, status, and JSON
error body (or raw response text when JSON is unavailable).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from index_job_service import _run_pipeline_step


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, text: str = "", json_error=None):
        self.status_code = status_code
        self._json_body = json_body
        self._text = text
        self._json_error = json_error

    def get_json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_body

    def get_data(self, as_text: bool = False):
        return self._text


def test_run_pipeline_step_success_does_not_raise() -> None:
    client = MagicMock()
    client.post.return_value = _FakeResponse(200, json_body={"ok": True})
    payload = {"database": "privacy-compliance"}
    _run_pipeline_step(client, "create-embeddings", "/create-embeddings", payload)
    client.post.assert_called_once_with("/create-embeddings", json=payload)


def test_run_pipeline_step_uses_json_error_body() -> None:
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        400, json_body={"error": "text_column must be a non-empty string"}
    )
    with pytest.raises(RuntimeError) as exc:
        _run_pipeline_step(
            client,
            "create-embeddings",
            "/create-embeddings",
            {"text_column": " "},
        )
    message = str(exc.value)
    assert "create-embeddings /create-embeddings failed (400)" in message
    assert "text_column must be a non-empty string" in message


def test_run_pipeline_step_falls_back_to_raw_body_when_json_missing() -> None:
    client = MagicMock()
    client.post.return_value = _FakeResponse(500, json_body=None, text="upstream exploded")
    with pytest.raises(RuntimeError) as exc:
        _run_pipeline_step(client, "create-vector-index", "/create-vector-index", {})
    message = str(exc.value)
    assert "create-vector-index /create-vector-index failed (500)" in message
    assert "upstream exploded" in message


def test_run_pipeline_step_falls_back_when_get_json_raises() -> None:
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        502, json_error=ValueError("not json"), text="bad gateway"
    )
    with pytest.raises(RuntimeError) as exc:
        _run_pipeline_step(client, "create-statute-subsections", "/create-statute-subsections", {})
    message = str(exc.value)
    assert "create-statute-subsections /create-statute-subsections failed (502)" in message
    assert "bad gateway" in message
