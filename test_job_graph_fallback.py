"""Regression tests for the fallback job graph and pipeline failure reporting.

Background compliance and index jobs compile a linear graph. When langgraph is
not installed they run on compliance_graph.CompiledGraph. These tests pin the
walk/merge rules that graph actually implements, and the job-node contracts
that decide whether a run is stored, completed, or left failed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from flask import Flask, jsonify, request

from compliance_graph import END, START, StateGraph
from compliance_job_service import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    build_gap_analysis_graph,
    build_health_score_graph,
)
from index_job_service import _run_pipeline_step, build_sub_vector_index_graph


def _run(coro):
    return asyncio.run(coro)


def test_fallback_graph_merges_sync_and_async_results() -> None:
    """Node dicts merge in order. Keys a node does not return stay put."""

    def mark_running(state: dict) -> dict:
        assert state["job_id"] == "job-1"
        assert state["request"] == {"policy_document_id": "pol-1"}
        return {"status": "running"}

    async def finish(state: dict) -> dict:
        assert state["status"] == "running"
        assert state["job_id"] == "job-1"
        return {"status": "completed", "result": {"ok": True}}

    graph = StateGraph(dict)
    graph.add_node("mark_running", mark_running)
    graph.add_node("finish", finish)
    graph.add_edge(START, "mark_running")
    graph.add_edge("mark_running", "finish")
    graph.add_edge("finish", END)

    out = _run(
        graph.compile().ainvoke(
            {"job_id": "job-1", "request": {"policy_document_id": "pol-1"}}
        )
    )
    assert out == {
        "job_id": "job-1",
        "request": {"policy_document_id": "pol-1"},
        "status": "completed",
        "result": {"ok": True},
    }


def test_fallback_graph_keeps_walking_when_a_node_returns_nothing() -> None:
    """None and non-dict returns do not replace state. In-place edits survive."""
    seen: dict[str, Any] = {}

    def mutate(state: dict) -> None:
        state["status"] = "running"
        return None

    def ignore(state: dict) -> str:
        assert state["status"] == "running"
        return "not-a-state"

    def finish(state: dict) -> dict:
        seen["status"] = state["status"]
        seen["job_id"] = state["job_id"]
        return {"status": "completed"}

    graph = StateGraph(dict)
    graph.add_node("mutate", mutate)
    graph.add_node("ignore", ignore)
    graph.add_node("finish", finish)
    graph.add_edge(START, "mutate")
    graph.add_edge("mutate", "ignore")
    graph.add_edge("ignore", "finish")
    graph.add_edge("finish", END)

    out = _run(graph.compile().ainvoke({"job_id": "job-1"}))
    assert seen == {"status": "running", "job_id": "job-1"}
    assert out["status"] == "completed"
    assert out["job_id"] == "job-1"


def test_fallback_graph_stops_on_errors_and_follows_only_the_first_edge() -> None:
    """A raised node aborts the walk. Extra edges and missing nodes are not taken."""
    called: list[str] = []

    def boom(_state: dict) -> dict:
        called.append("boom")
        raise RuntimeError("db down")

    def later(_state: dict) -> dict:
        called.append("later")
        return {"status": "completed"}

    graph = StateGraph(dict)
    graph.add_node("boom", boom)
    graph.add_node("later", later)
    graph.add_edge(START, "boom")
    graph.add_edge("boom", "later")
    graph.add_edge("later", END)
    with pytest.raises(RuntimeError, match="db down"):
        _run(graph.compile().ainvoke({"job_id": "job-1"}))
    assert called == ["boom"]

    def first(_state: dict) -> dict:
        called.append("first")
        return {"took": "first"}

    def second(_state: dict) -> dict:
        called.append("second")
        return {"took": "second"}

    called.clear()
    branched = StateGraph(dict)
    branched.add_node("first", first)
    branched.add_node("second", second)
    branched.add_edge(START, "first")
    branched.add_edge("first", "second")
    branched.add_edge("first", "not-a-node")
    out = _run(branched.compile().ainvoke({"job_id": "job-1"}))
    assert called == ["first", "second"]
    assert out["took"] == "second"
    assert out["job_id"] == "job-1"

    called.clear()
    missing = StateGraph(dict)
    missing.add_node("second", second)
    missing.add_edge(START, "missing")
    missing.add_edge("missing", "second")
    out = _run(missing.compile().ainvoke({"job_id": "job-9"}))
    assert called == []
    assert out == {"job_id": "job-9"}

    idle = StateGraph(dict)
    idle.add_node("first", first)
    out = _run(idle.compile().ainvoke({"job_id": "job-2", "request": {}}))
    assert called == []
    assert out == {"job_id": "job-2", "request": {}}


def _gap_result(**overrides: Any) -> SimpleNamespace:
    payload = {
        "policy_document_id": "pol-1",
        "company_name": "Acme",
        "applicable_jurisdictions": ["CA"],
        "gaps": [{"status": "missing"}],
        "summary": {"total_requirements": 1},
        "analyzed_at": "2026-01-01T00:00:00+00:00",
        "run_types": ["gap_v4"],
        "run_type": "gap_analysis_v4",
        "version": "v4",
        "retrieval_metadata": {"statute_items_considered": 2},
        "statute_chunk_ids_used": ["chunk-1"],
    }
    payload.update(overrides)
    return SimpleNamespace(model_dump=lambda: payload)


def test_gap_job_persists_result_and_keeps_falsy_fields_from_being_copied() -> None:
    """A completed gap job writes the result, then the run log, and merges state.

    Empty run_types is stored as ["gap"]. version 0, run_type "", an empty
    retrieval_metadata, and an empty summary are omitted rather than stored.
    """
    jobs = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    storage.write_compliance_run_log = AsyncMock()
    seen: dict[str, Any] = {}

    async def run_gap(req: Any) -> SimpleNamespace:
        seen["policy_document_id"] = req.policy_document_id
        return _gap_result(
            run_types=[],
            run_type="",
            version=0,
            retrieval_metadata={},
            summary={},
            statute_chunk_ids_used=None,
            company_name=None,
        )

    graph = build_gap_analysis_graph(run_gap, jobs, storage)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-1",
                "job_type": "gap_analysis",
                "status": "pending",
                "request": {"policy_document_id": "pol-1"},
            }
        )
    )

    assert seen == {"policy_document_id": "pol-1"}
    assert out["job_id"] == "job-1"
    assert out["request"] == {"policy_document_id": "pol-1"}
    assert out["status"] == JOB_STATUS_COMPLETED
    assert out["started_at"]
    assert out["completed_at"]
    assert out["result"]["policy_document_id"] == "pol-1"

    statuses = [call.args[1] for call in jobs.update_job_status.call_args_list]
    assert statuses == [JOB_STATUS_RUNNING, JOB_STATUS_COMPLETED]
    assert jobs.update_job_status.call_args_list[1].kwargs["result"]["gaps"] == [
        {"status": "missing"}
    ]

    res_doc = storage.write_compliance_result.await_args.args[0]
    assert res_doc["policy_document_id"] == "pol-1"
    assert res_doc["company_name"] is None
    assert res_doc["applicable_jurisdictions"] == ["CA"]
    assert res_doc["run_types"] == ["gap"]
    assert "run_type" not in res_doc
    assert "version" not in res_doc
    assert "retrieval_metadata" not in res_doc

    log_doc = storage.write_compliance_run_log.await_args.args[0]
    assert log_doc["policy_document_id"] == "pol-1"
    assert log_doc["applicable_jurisdictions"] == ["CA"]
    assert log_doc["run_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert log_doc["statute_chunk_ids_used"] == []
    assert "summary" not in log_doc
    assert "run_type" not in log_doc


def test_gap_job_completes_when_result_persistence_fails() -> None:
    """A Mongo write error does not fail the job and does not write the run log."""
    jobs = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock(side_effect=RuntimeError("mongo down"))
    storage.write_compliance_run_log = AsyncMock()

    async def run_gap(_req: Any) -> SimpleNamespace:
        return _gap_result()

    graph = build_gap_analysis_graph(run_gap, jobs, storage)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-2",
                "status": "pending",
                "request": {"policy_document_id": "pol-1"},
            }
        )
    )

    assert out["status"] == JOB_STATUS_COMPLETED
    assert out["result"]["run_type"] == "gap_analysis_v4"
    storage.write_compliance_run_log.assert_not_called()
    completed = jobs.update_job_status.call_args_list[-1]
    assert completed.args[1] == JOB_STATUS_COMPLETED
    assert completed.kwargs["result"]["policy_document_id"] == "pol-1"


def test_gap_job_validation_failure_does_not_run_or_persist() -> None:
    """An invalid job payload is marked failed before the analyzer or storage."""
    jobs = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    called = False

    async def run_gap(_req: Any) -> SimpleNamespace:
        nonlocal called
        called = True
        return _gap_result()

    graph = build_gap_analysis_graph(run_gap, jobs, storage)
    out = _run(
        graph.ainvoke(
            {"job_id": "job-3", "status": "pending", "request": {}}
        )
    )

    assert called is False
    storage.write_compliance_result.assert_not_called()
    assert out["status"] == JOB_STATUS_FAILED
    assert "policy_document_id" in out["error"]
    failed = jobs.update_job_status.call_args_list[-1]
    assert failed.args == ("job-3", JOB_STATUS_FAILED)
    assert "policy_document_id" in failed.kwargs["error"]


def test_health_score_job_stores_zero_and_survives_a_write_error() -> None:
    """Score 0 and components None are stored. A write error still completes the job."""
    jobs = MagicMock()
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()

    async def run_score(req: Any) -> SimpleNamespace:
        assert req.policy_document_id == "pol-1"
        return SimpleNamespace(
            model_dump=lambda: {
                "policy_document_id": "pol-1",
                "company_name": "",
                "applicable_jurisdictions": ["CA"],
                "privacy_health_score": 0,
                "score_assessment": "Critical.",
                "score_breakdown": {},
                "components": None,
                "analyzed_at": "2026-01-02T00:00:00+00:00",
            }
        )

    graph = build_health_score_graph(run_score, jobs, storage)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-h",
                "status": "pending",
                "request": {"policy_document_id": "pol-1"},
            }
        )
    )
    assert out["status"] == JOB_STATUS_COMPLETED
    res_doc = storage.write_compliance_result.await_args.args[0]
    assert res_doc["privacy_health_score"] == 0
    assert res_doc["company_name"] == ""
    assert res_doc["score_breakdown"] == {}
    assert res_doc["components"] is None
    assert res_doc["run_types"] == ["health_score"]
    assert res_doc["score_assessment"] == "Critical."

    storage.write_compliance_result = AsyncMock(side_effect=RuntimeError("mongo down"))
    graph = build_health_score_graph(run_score, jobs, storage)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-h2",
                "status": "pending",
                "request": {"policy_document_id": "pol-1"},
            }
        )
    )
    assert out["status"] == JOB_STATUS_COMPLETED
    assert out["result"]["privacy_health_score"] == 0
    assert jobs.update_job_status.call_args_list[-1].args[1] == JOB_STATUS_COMPLETED


def _policy_index_app(captured: list[Any], response_body: Any, status_code: int) -> Flask:
    app = Flask(__name__)

    @app.post("/create-embeddings")
    def create_embeddings():
        captured.append(request.get_json(silent=True))
        if isinstance(response_body, dict):
            return jsonify(response_body), status_code
        return response_body, status_code

    @app.post("/create-vector-index")
    def create_vector_index():
        captured.append("vector-index")
        return jsonify({"ok": True})

    return app


def test_index_job_stores_pipeline_http_error_and_stops() -> None:
    """A non-2xx pipeline step fails the job with the endpoint error and skips later steps."""
    captured: list[Any] = []
    app = _policy_index_app(captured, {"error": "no model"}, 400)
    jobs = MagicMock()
    jobs._config = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks",
        policy_legal_embeddings_collection="policy_legal_embeddings",
    )
    graph = build_sub_vector_index_graph(app, jobs)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-i",
                "status": "pending",
                "request": {
                    "document_type": "policy",
                    "source_query": {"document_id": "doc-1"},
                },
            }
        )
    )

    assert out["status"] == JOB_STATUS_FAILED
    assert out["error"] == "create-embeddings /create-embeddings failed (400): no model"
    assert out["started_at"]
    assert "vector-index" not in captured
    assert captured[0]["source_database_name"] == "privacy-compliance"
    assert captured[0]["source_collection_name"] == "policy_chunks"
    assert captured[0]["index_collection_name"] == "policy_legal_embeddings"
    assert captured[0]["source_query"] == {"document_id": "doc-1"}
    assert captured[0]["text_column"] == "chunk_text"
    failed = jobs.update_job_status.call_args_list[-1]
    assert failed.args[1] == JOB_STATUS_FAILED
    assert failed.kwargs["error"] == out["error"]


def test_index_job_false_source_query_is_posted_as_an_empty_object() -> None:
    """A falsy source_query is replaced with {} before the embeddings call."""
    captured: list[Any] = []
    app = _policy_index_app(captured, "upstream down", 500)
    jobs = MagicMock()
    jobs._config = SimpleNamespace(
        compliance_database="privacy-compliance",
        policy_chunks_collection="policy_chunks",
        policy_legal_embeddings_collection="policy_legal_embeddings",
    )
    graph = build_sub_vector_index_graph(app, jobs)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-f",
                "status": "pending",
                "request": {"document_type": "policy", "source_query": False},
            }
        )
    )
    assert captured[0]["source_query"] == {}
    assert out["error"] == "create-embeddings /create-embeddings failed (500): upstream down"
    assert "vector-index" not in captured


def test_index_job_rejects_unknown_document_type_after_marking_running() -> None:
    """document_type is case-sensitive. The check runs after the app context opens.

    The failed state still keeps started_at from the validate node.
    """
    app = Flask(__name__)
    jobs = MagicMock()
    jobs._config = SimpleNamespace(compliance_database="privacy-compliance")
    graph = build_sub_vector_index_graph(app, jobs)
    out = _run(
        graph.ainvoke(
            {
                "job_id": "job-t",
                "status": "pending",
                "request": {"document_type": "Policy"},
            }
        )
    )
    assert out["status"] == JOB_STATUS_FAILED
    assert out["error"] == "Invalid document_type: Policy"
    assert out["started_at"]
    assert out["completed_at"]
    statuses = [call.args[1] for call in jobs.update_job_status.call_args_list]
    assert statuses == [JOB_STATUS_RUNNING, JOB_STATUS_FAILED]


class _Resp:
    def __init__(
        self,
        status_code: int,
        json_body: Any = None,
        text: str = "",
        json_error: BaseException | None = None,
        data_error: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_body = json_body
        self._text = text
        self._json_error = json_error
        self._data_error = data_error

    def get_json(self) -> Any:
        if self._json_error:
            raise self._json_error
        return self._json_body

    def get_data(self, as_text: bool = False) -> str:
        if self._data_error:
            raise self._data_error
        return self._text


class _Client:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp
        self.calls: list[tuple[str, Any]] = []

    def post(self, path: str, json: Any = None) -> _Resp:
        self.calls.append((path, json))
        return self._resp


def test_pipeline_step_error_text_prefers_json_then_body_then_status() -> None:
    """Job error strings follow the helper's fallback order, including its quirks."""
    ok = _Client(_Resp(200, {"ok": True}))
    assert _run_pipeline_step(ok, "create-embeddings", "/create-embeddings", {"q": 1}) is None
    assert ok.calls == [("/create-embeddings", {"q": 1})]

    almost = _Client(_Resp(399, {"error": "nope"}, text="body"))
    assert _run_pipeline_step(almost, "create-embeddings", "/create-embeddings", {}) is None

    json_error = _Client(_Resp(400, {"error": "no model"}, text="ignored"))
    with pytest.raises(
        RuntimeError,
        match=r"create-embeddings /create-embeddings failed \(400\): no model",
    ):
        _run_pipeline_step(json_error, "create-embeddings", "/create-embeddings", {})

    blank_error = _Client(_Resp(400, {"error": ""}, text="body wins if default is used"))
    with pytest.raises(RuntimeError, match=r"failed \(400\): $"):
        _run_pipeline_step(blank_error, "create-embeddings", "/create-embeddings", {})

    missing_key = _Client(_Resp(400, {}, text="plain body"))
    with pytest.raises(RuntimeError, match=r"failed \(400\): plain body$"):
        _run_pipeline_step(missing_key, "create-embeddings", "/create-embeddings", {})

    list_body = _Client(_Resp(400, ["not", "an", "object"], text="from body"))
    with pytest.raises(RuntimeError, match=r"failed \(400\): from body$"):
        _run_pipeline_step(list_body, "create-embeddings", "/create-embeddings", {})

    empty = _Client(_Resp(502, None, text=""))
    with pytest.raises(RuntimeError, match=r"failed \(502\): 502$"):
        _run_pipeline_step(empty, "create-embeddings", "/create-embeddings", {})

    broken = _Client(
        _Resp(
            400,
            {"error": "no model"},
            data_error=RuntimeError("body unreadable"),
        )
    )
    with pytest.raises(RuntimeError, match="body unreadable"):
        _run_pipeline_step(broken, "create-embeddings", "/create-embeddings", {})
