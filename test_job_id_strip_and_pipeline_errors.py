"""Regression tests for job GET JSON-safety and pipeline HTTP error mapping.

GET /jobs and GET /index-jobs jsonify storage documents. Mongo `_id` must be
stripped or Flask returns 500. Background index jobs turn non-2xx pipeline
steps into RuntimeError so the job is marked failed with a usable message.
"""
from __future__ import annotations

import pytest
from flask import Flask, jsonify
from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_job_service import ComplianceJobStorage
from index_job_service import IndexJobStorage, _run_pipeline_step


def _storage_coll(storage_cls):
    mock_mongo = MagicMock()
    mock_coll = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_coll
    storage = storage_cls(mock_mongo, load_config())
    return storage, mock_coll


@pytest.mark.parametrize("storage_cls", [ComplianceJobStorage, IndexJobStorage])
def test_get_job_strips_mongo_id_so_json_responses_stay_serializable(storage_cls) -> None:
    """Both job stores delete `_id` before returning the document to HTTP handlers."""
    storage, mock_coll = _storage_coll(storage_cls)
    mock_coll.find_one.return_value = {
        "_id": object(),
        "job_id": "job-1",
        "status": "completed",
        "result": {"ok": True},
    }
    retrieved = storage.get_job("job-1")
    assert retrieved is not None
    assert retrieved["job_id"] == "job-1"
    assert retrieved["status"] == "completed"
    assert "_id" not in retrieved
    mock_coll.find_one.assert_called_once_with({"job_id": "job-1"})


def test_run_pipeline_step_success_does_not_raise() -> None:
    app = Flask(__name__)

    @app.post("/ok")
    def ok():
        return jsonify({"status": "ok"}), 200

    with app.test_client() as client:
        _run_pipeline_step(client, "create-embeddings", "/ok", {"k": "v"})


def test_run_pipeline_step_includes_json_error_in_runtimeerror() -> None:
    app = Flask(__name__)

    @app.post("/fail")
    def fail():
        return jsonify({"error": "source collection missing"}), 400

    with app.test_client() as client:
        with pytest.raises(RuntimeError) as exc:
            _run_pipeline_step(client, "create-chunks", "/fail", {})
    message = str(exc.value)
    assert "create-chunks /fail failed (400)" in message
    assert "source collection missing" in message


def test_run_pipeline_step_falls_back_to_raw_body_when_json_has_no_error() -> None:
    app = Flask(__name__)

    @app.post("/plain")
    def plain():
        return "upstream exploded", 502

    with app.test_client() as client:
        with pytest.raises(RuntimeError) as exc:
            _run_pipeline_step(client, "create-vector-index", "/plain", {"x": 1})
    message = str(exc.value)
    assert "create-vector-index /plain failed (502)" in message
    assert "upstream exploded" in message
