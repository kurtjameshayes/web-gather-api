"""Regression tests for background job graphs.

Covers the sub-vector-index step sequence, gap-analysis and health-score
persistence, and the thread wrapper that records graph crashes as failed jobs.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from flask import Flask, jsonify, request
from unittest.mock import patch

from compliance_config import load_config
from compliance_job_service import (
    JOB_TYPE_GAP_ANALYSIS,
    JOB_TYPE_HEALTH_SCORE,
    build_gap_analysis_graph,
    build_health_score_graph,
    start_gap_analysis_job,
    start_health_score_job,
)
from index_job_service import (
    JOB_TYPE_SUB_VECTOR_INDEX,
    build_sub_vector_index_graph,
    start_sub_vector_index_job,
)


class _JobStore:
    """Records status updates without MongoDB."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self._config = load_config()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.updates.append(
            {"job_id": job_id, "status": status, "result": result, "error": error}
        )


class _ResultStorage:
    """Async stand-in for ComplianceStorage writes."""

    def __init__(self, fail_result: bool = False) -> None:
        self.results: list[dict[str, Any]] = []
        self.logs: list[dict[str, Any]] = []
        self.fail_result = fail_result

    async def write_compliance_result(self, doc: dict[str, Any]) -> str:
        if self.fail_result:
            raise RuntimeError("mongo down")
        self.results.append(doc)
        return "result-id"

    async def write_compliance_run_log(self, doc: dict[str, Any]) -> None:
        self.logs.append(doc)


class _ImmediateThread:
    """Runs the worker target on start() so tests stay single-threaded."""

    def __init__(self, target: Any = None, daemon: Any = None, **_kwargs: Any) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


def _recording_app(handlers: dict[str, tuple[int, Any]]) -> tuple[Flask, list[tuple[str, Any]]]:
    app = Flask(__name__)
    calls: list[tuple[str, Any]] = []

    @app.post("/<path:subpath>")
    def handle(subpath: str):
        path = "/" + subpath
        calls.append((path, request.get_json(silent=True)))
        spec = handlers.get(path)
        if spec is None:
            return jsonify({"error": f"unexpected {path}"}), 404
        status, payload = spec
        if isinstance(payload, str):
            return payload, status
        return jsonify(payload), status

    return app, calls


def _run_index(app: Flask, store: _JobStore, request_dict: dict[str, Any]) -> dict[str, Any]:
    graph = build_sub_vector_index_graph(app, store)
    return asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-index",
                "job_type": JOB_TYPE_SUB_VECTOR_INDEX,
                "status": "pending",
                "request": request_dict,
            }
        )
    )


def test_statute_pipeline_posts_four_steps_and_embeds_sub_chunks() -> None:
    """Statute jobs embed statute_sub_chunks, not the subtopics collection."""
    cfg = load_config()
    db = cfg.compliance_database
    query = {"document_id": "stat-1"}
    app, calls = _recording_app(
        {
            "/create-statute-subsections": (200, {}),
            "/create-statute-subtopics": (200, {}),
            "/create-embeddings": (200, {}),
            "/create-vector-index": (200, {}),
        }
    )
    store = _JobStore()

    state = _run_index(app, store, {"document_type": "statute", "source_query": query})

    assert [path for path, _body in calls] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
        "/create-embeddings",
        "/create-vector-index",
    ]
    assert calls[0][1] == {
        "column": "chunk_text",
        "database": db,
        "destination_collection": "statute_sub_chunks",
        "parse_prompt": "",
        "source_collection": "statute_chunks",
        "source_query": query,
        "subsection_column": "sub_chunk_text",
    }
    assert calls[1][1] == {
        "column": "sub_chunk_text",
        "database": db,
        "destination_collection": "statute_subtopics",
        "parse_prompt": "",
        "source_collection": "statute_sub_chunks",
        "source_query": query,
        "subsection_column": "sub_chunk_text",
    }
    assert calls[2][1] == {
        "index_collection_name": "statute_sub_embeddings",
        "index_database_name": db,
        "source_collection_name": "statute_sub_chunks",
        "source_database_name": db,
        "source_query": query,
        "text_column": "sub_chunk_text",
    }
    assert calls[3][1] == {
        "collection_name": "statute_sub_embeddings",
        "database_name": db,
        "filter_fields": ["jurisdiction", "document_id"],
        "index_name": "vector_index",
    }
    assert state["status"] == "completed"
    assert state["job_id"] == "job-index"
    assert state["started_at"]
    assert state["result"] == {
        "document_type": "statute",
        "source_query": query,
        "status": "completed",
    }
    assert [item["status"] for item in store.updates] == ["running", "completed"]
    assert store.updates[1]["result"] == state["result"]
    assert store.updates[1]["error"] is None


def test_policy_pipeline_embeds_configured_policy_collections() -> None:
    """Policy jobs skip statute endpoints and use the configured collection names."""
    cfg = load_config()
    db = cfg.compliance_database
    app, calls = _recording_app(
        {
            "/create-embeddings": (200, {}),
            "/create-vector-index": (200, {}),
        }
    )
    store = _JobStore()

    state = _run_index(app, store, {"document_type": "policy"})

    assert calls == [
        (
            "/create-embeddings",
            {
                "source_database_name": db,
                "source_collection_name": cfg.policy_chunks_collection,
                "index_database_name": db,
                "index_collection_name": cfg.policy_legal_embeddings_collection,
                "source_query": {},
                "text_column": "chunk_text",
            },
        ),
        (
            "/create-vector-index",
            {
                "collection_name": cfg.policy_legal_embeddings_collection,
                "database_name": db,
                "filter_fields": ["document_id"],
                "index_name": "vector_index",
            },
        ),
    ]
    assert state["status"] == "completed"
    assert state["result"]["source_query"] == {}


def test_pipeline_stops_on_http_error_and_formats_error_bodies() -> None:
    """A non-2xx step fails the job, skips later steps, and keeps the error text."""
    app, calls = _recording_app(
        {
            "/create-statute-subsections": (200, {}),
            "/create-statute-subtopics": (400, {"error": "llm down"}),
        }
    )
    store = _JobStore()
    state = _run_index(
        app, store, {"document_type": "statute", "source_query": {"document_id": "s1"}}
    )

    assert [path for path, _body in calls] == [
        "/create-statute-subsections",
        "/create-statute-subtopics",
    ]
    assert state["status"] == "failed"
    assert state["error"] == (
        "create-statute-subtopics /create-statute-subtopics failed (400): llm down"
    )
    assert store.updates[-1]["status"] == "failed"
    assert store.updates[-1]["result"] is None
    assert "completed" not in [item["status"] for item in store.updates]

    plain_app, _plain_calls = _recording_app(
        {"/create-embeddings": (502, "plain failure")}
    )
    plain_store = _JobStore()
    plain_state = _run_index(plain_app, plain_store, {"document_type": "policy"})
    assert plain_state["error"] == (
        "create-embeddings /create-embeddings failed (502): plain failure"
    )

    raw_app, _raw_calls = _recording_app(
        {"/create-embeddings": (400, {"message": "nope"})}
    )
    raw_state = _run_index(raw_app, _JobStore(), {"document_type": "policy"})
    assert raw_state["error"].startswith(
        "create-embeddings /create-embeddings failed (400): "
    )
    assert '"message"' in raw_state["error"]
    assert not raw_state["error"].endswith(": nope")

    empty_app, _empty_calls = _recording_app({"/create-embeddings": (500, "")})
    empty_state = _run_index(empty_app, _JobStore(), {"document_type": "policy"})
    assert empty_state["error"] == (
        "create-embeddings /create-embeddings failed (500): 500"
    )


def test_pipeline_rejects_unnormalized_document_type_before_http() -> None:
    """The worker compares document_type exactly and coerces a null query to {}."""
    app, calls = _recording_app({})
    store = _JobStore()

    state = _run_index(
        app, store, {"document_type": "Statute", "source_query": None}
    )

    assert calls == []
    assert state["status"] == "failed"
    assert state["error"] == "Invalid document_type: Statute"
    assert state["started_at"]
    assert store.updates[-1]["error"] == "Invalid document_type: Statute"


def _run_gap(store: _JobStore, request_dict: dict[str, Any], runner: Any, storage: Any) -> dict[str, Any]:
    graph = build_gap_analysis_graph(runner, store, storage)
    return asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-gap",
                "job_type": JOB_TYPE_GAP_ANALYSIS,
                "status": "pending",
                "request": request_dict,
            }
        )
    )


def test_gap_job_defaults_falsy_run_fields_and_skips_empty_summary() -> None:
    """Empty run_types becomes ['gap']; empty summary and metadata are omitted."""
    store = _JobStore()
    storage = _ResultStorage()

    async def runner(req: Any) -> dict[str, Any]:
        assert req.policy_document_id == "pol-1"
        return {
            "policy_document_id": "pol-1",
            "company_name": "Acme",
            "gaps": [],
            "summary": {},
            "analyzed_at": "2026-01-01T00:00:00+00:00",
            "run_types": [],
            "run_type": "",
            "version": None,
            "retrieval_metadata": {},
            "statute_chunk_ids_used": None,
        }

    state = _run_gap(store, {"policy_document_id": "pol-1"}, runner, storage)

    assert state["status"] == "completed"
    assert storage.results == [
        {
            "policy_document_id": "pol-1",
            "company_name": "Acme",
            "applicable_jurisdictions": [],
            "gaps": [],
            "summary": {},
            "analyzed_at": "2026-01-01T00:00:00+00:00",
            "run_types": ["gap"],
        }
    ]
    assert storage.logs == [
        {
            "policy_document_id": "pol-1",
            "applicable_jurisdictions": [],
            "run_timestamp": "2026-01-01T00:00:00+00:00",
            "statute_chunk_ids_used": [],
        }
    ]


def test_gap_job_persists_model_dump_optional_fields() -> None:
    """Objects with model_dump() supply run_type, version, metadata, and chunk ids."""
    store = _JobStore()
    storage = _ResultStorage()

    class _Dumpable:
        def model_dump(self) -> dict[str, Any]:
            return {
                "policy_document_id": "pol-9",
                "company_name": "Acme",
                "applicable_jurisdictions": ["CA"],
                "gaps": [{"status": "missing"}],
                "summary": {"conflicts": 1},
                "analyzed_at": "2026-01-02T00:00:00+00:00",
                "run_types": ["gap_v4"],
                "run_type": "gap_analysis_v4",
                "version": "v4",
                "retrieval_metadata": {"top_k": 5},
                "statute_chunk_ids_used": ["c1"],
            }

    async def runner(_req: Any) -> _Dumpable:
        return _Dumpable()

    state = _run_gap(store, {"policy_document_id": "pol-9"}, runner, storage)

    assert state["status"] == "completed"
    assert state["result"]["version"] == "v4"
    written = storage.results[0]
    assert written["run_types"] == ["gap_v4"]
    assert written["run_type"] == "gap_analysis_v4"
    assert written["version"] == "v4"
    assert written["retrieval_metadata"] == {"top_k": 5}
    assert storage.logs[0]["statute_chunk_ids_used"] == ["c1"]
    assert storage.logs[0]["run_type"] == "gap_analysis_v4"
    assert storage.logs[0]["summary"] == {"conflicts": 1}


def test_gap_job_completes_when_result_write_fails_and_skips_the_run_log() -> None:
    """A storage error is logged and does not fail a finished analysis."""
    store = _JobStore()
    storage = _ResultStorage(fail_result=True)

    async def runner(_req: Any) -> dict[str, Any]:
        return {"policy_document_id": "pol-1", "analyzed_at": "t", "gaps": []}

    state = _run_gap(store, {"policy_document_id": "pol-1"}, runner, storage)

    assert state["status"] == "completed"
    assert state["result"]["policy_document_id"] == "pol-1"
    assert storage.logs == []
    assert store.updates[-1]["status"] == "completed"
    assert store.updates[-1]["error"] is None


def test_gap_job_without_storage_still_completes() -> None:
    """save path is optional; a missing storage client does not fail the job."""
    store = _JobStore()

    async def runner(_req: Any) -> dict[str, Any]:
        return {"policy_document_id": "pol-1", "gaps": [1]}

    state = _run_gap(store, {"policy_document_id": "pol-1"}, runner, None)

    assert state["status"] == "completed"
    assert store.updates[-1]["result"]["gaps"] == [1]


def test_gap_job_failure_does_not_persist() -> None:
    """Analysis exceptions mark the job failed and skip compliance writes."""
    store = _JobStore()
    storage = _ResultStorage()

    async def runner(_req: Any) -> dict[str, Any]:
        raise RuntimeError("llm exploded")

    state = _run_gap(store, {"policy_document_id": "pol-1"}, runner, storage)

    assert state["status"] == "failed"
    assert state["error"] == "llm exploded"
    assert storage.results == []
    assert storage.logs == []
    assert store.updates[-1]["result"] is None


def test_gap_job_rejects_whitespace_policy_id_before_the_runner() -> None:
    """A blank policy id fails validation and never calls gap analysis."""
    store = _JobStore()
    storage = _ResultStorage()
    called = False

    async def runner(_req: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    state = _run_gap(store, {"policy_document_id": "   "}, runner, storage)

    assert called is False
    assert state["status"] == "failed"
    assert "must be provided" in state["error"]
    assert storage.results == []


def _run_health(store: _JobStore, request_dict: dict[str, Any], runner: Any, storage: Any) -> dict[str, Any]:
    graph = build_health_score_graph(runner, store, storage)
    return asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-health",
                "job_type": JOB_TYPE_HEALTH_SCORE,
                "status": "pending",
                "request": request_dict,
            }
        )
    )


def test_health_job_persists_score_fields_and_forces_health_score_run_type() -> None:
    """Health jobs do not write a run log and ignore a caller-supplied run_types."""
    store = _JobStore()
    storage = _ResultStorage()

    async def runner(_req: Any) -> dict[str, Any]:
        return {
            "policy_document_id": "pol-1",
            "company_name": None,
            "privacy_health_score": 80,
            "score_assessment": "Good",
            "analyzed_at": "2026-01-03T00:00:00+00:00",
            "run_types": ["gap"],
        }

    state = _run_health(store, {"policy_document_id": "pol-1"}, runner, storage)

    assert state["status"] == "completed"
    assert storage.logs == []
    assert storage.results == [
        {
            "policy_document_id": "pol-1",
            "company_name": None,
            "applicable_jurisdictions": [],
            "privacy_health_score": 80,
            "score_assessment": "Good",
            "score_breakdown": {},
            "components": {},
            "analyzed_at": "2026-01-03T00:00:00+00:00",
            "run_types": ["health_score"],
        }
    ]


def test_health_job_completes_when_result_write_fails() -> None:
    """A storage error does not flip a finished health-score job to failed."""
    store = _JobStore()
    storage = _ResultStorage(fail_result=True)

    async def runner(_req: Any) -> dict[str, Any]:
        return {"policy_document_id": "pol-1", "privacy_health_score": 10, "analyzed_at": "t"}

    state = _run_health(store, {"policy_document_id": "pol-1"}, runner, storage)

    assert state["status"] == "completed"
    assert storage.logs == []
    assert store.updates[-1]["status"] == "completed"


def test_health_job_rejects_empty_policy_id_before_the_runner() -> None:
    """An empty policy id fails schema validation and does not score."""
    store = _JobStore()
    called = False

    async def runner(_req: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    state = _run_health(store, {"policy_document_id": ""}, runner, _ResultStorage())

    assert called is False
    assert state["status"] == "failed"
    assert store.updates[-1]["error"]


@pytest.mark.parametrize(
    ("module_name", "starter_name", "job_type", "request_dict"),
    [
        (
            "compliance_job_service",
            "start_gap_analysis_job",
            JOB_TYPE_GAP_ANALYSIS,
            {"policy_document_id": "pol-1"},
        ),
        (
            "compliance_job_service",
            "start_health_score_job",
            JOB_TYPE_HEALTH_SCORE,
            {"policy_document_id": "pol-1"},
        ),
        (
            "index_job_service",
            "start_sub_vector_index_job",
            JOB_TYPE_SUB_VECTOR_INDEX,
            {"document_type": "policy", "source_query": {}},
        ),
    ],
)
def test_background_starter_records_graph_crash_and_still_returns_job_id(
    module_name: str,
    starter_name: str,
    job_type: str,
    request_dict: dict[str, Any],
) -> None:
    """A crash outside the node handler is stored as failed and is not re-raised."""
    calls: list[tuple[str, Optional[str]]] = []
    store = _JobStore()

    def create_job(created_type: str, created_request: dict[str, Any]) -> str:
        assert created_type == job_type
        assert created_request == request_dict
        return "job-1"

    def update_job_status(job_id: str, status: str, result: Any = None, error: Optional[str] = None) -> None:
        calls.append((status, error))
        if status == "running":
            raise RuntimeError("storage down")

    store.create_job = create_job  # type: ignore[method-assign]
    store.update_job_status = update_job_status  # type: ignore[method-assign]

    starters = {
        "start_gap_analysis_job": lambda: start_gap_analysis_job(
            request_dict, store, _unused_runner
        ),
        "start_health_score_job": lambda: start_health_score_job(
            request_dict, store, _unused_runner
        ),
        "start_sub_vector_index_job": lambda: start_sub_vector_index_job(
            request_dict, store, Flask(__name__)
        ),
    }

    async def _unused_runner(_req: Any) -> dict[str, Any]:
        raise AssertionError("runner should not be called")

    with patch(f"{module_name}.threading.Thread", _ImmediateThread):
        job_id = starters[starter_name]()

    assert job_id == "job-1"
    assert calls == [("running", None), ("failed", "storage down")]
