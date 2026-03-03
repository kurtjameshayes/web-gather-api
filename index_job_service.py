"""Background job service for sub-vector-index workflow using LangGraph and MongoDB job storage."""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any, Dict, Optional, TypedDict

from compliance_config import ComplianceConfig
from compliance_utils import utc_now

logger = logging.getLogger("web-gather-api")

# Job status values
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

# Job type for sub-vector-index workflow
JOB_TYPE_SUB_VECTOR_INDEX = "sub_vector_index"

# Document types
DOCUMENT_TYPE_STATUTE = "statute"
DOCUMENT_TYPE_POLICY = "policy"


def _iso(dt=None) -> str:
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


class IndexJobState(TypedDict, total=False):
    """State for LangGraph sub-vector-index job workflow."""

    job_id: str
    job_type: str
    status: str
    request: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class IndexJobStorage:
    """MongoDB storage for index background jobs (index_job collection)."""

    def __init__(self, mongo_client: Any, config: ComplianceConfig) -> None:
        self._client = mongo_client
        self._config = config
        self._coll_name = config.index_jobs_collection
        self._db_name = config.compliance_database

    def _coll(self):
        return self._client[self._db_name][self._coll_name]

    def create_job(self, job_type: str, request: Dict[str, Any]) -> str:
        """Create a new job document and return job_id."""
        job_id = str(uuid.uuid4())
        doc = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JOB_STATUS_PENDING,
            "request": request,
            "result": None,
            "error": None,
            "created_at": _iso(),
            "started_at": None,
            "completed_at": None,
        }
        self._coll().insert_one(doc)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by job_id. Returns None if not found."""
        doc = self._coll().find_one({"job_id": job_id})
        if not doc:
            return None
        if "_id" in doc:
            del doc["_id"]
        return doc

    def update_job_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update job status and optionally result/error."""
        update: Dict[str, Any] = {"status": status}
        if status == JOB_STATUS_RUNNING:
            update["started_at"] = _iso()
        elif status in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED):
            update["completed_at"] = _iso()
        if result is not None:
            update["result"] = result
        if error is not None:
            update["error"] = error
        self._coll().update_one({"job_id": job_id}, {"$set": update})


def _run_pipeline_step(client, method: str, path: str, json_payload: Dict[str, Any]) -> None:
    """POST to an endpoint via test client; raise on non-2xx."""
    resp = client.post(path, json=json_payload)
    if resp.status_code >= 400:
        try:
            err_body = resp.get_json() or {}
            err_msg = err_body.get("error", resp.get_data(as_text=True) or str(resp.status_code))
        except Exception:
            err_msg = resp.get_data(as_text=True) or str(resp.status_code)
        raise RuntimeError(f"{method} {path} failed ({resp.status_code}): {err_msg}")


def build_sub_vector_index_graph(
    flask_app: Any,
    job_storage: IndexJobStorage,
) -> Any:
    """Build LangGraph StateGraph for sub-vector-index job. Falls back to compliance_graph if langgraph not installed."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        from compliance_graph import END, START, StateGraph

    def validate_and_start(state: IndexJobState) -> IndexJobState:
        """Validate request and mark job as running."""
        job_id = state["job_id"]
        job_storage.update_job_status(job_id, JOB_STATUS_RUNNING)
        return {
            "status": JOB_STATUS_RUNNING,
            "started_at": _iso(),
        }

    async def run_pipeline_node(state: IndexJobState) -> IndexJobState:
        """Execute subsections -> embeddings -> vector-index pipeline via Flask test client."""
        job_id = state["job_id"]
        request_dict = state.get("request") or {}
        document_type = request_dict.get("document_type")
        source_query = request_dict.get("source_query") or {}

        try:
            with flask_app.app_context():
                client = flask_app.test_client()

                db = "privacy-compliance"

                if document_type == DOCUMENT_TYPE_STATUTE:
                    # 1. create-statute-subsections
                    _run_pipeline_step(
                        client,
                        "create-statute-subsections",
                        "/create-statute-subsections",
                        {
                            "column": "chunk_text",
                            "database": db,
                            "destination_collection": "statute_sub_chunks",
                            "parse_prompt": "",
                            "source_collection": "statute_chunks",
                            "source_query": source_query,
                            "subsection_column": "sub_chunk_text",
                        },
                    )
                    # 2. create-statute-subtopics
                    _run_pipeline_step(
                        client,
                        "create-statute-subtopics",
                        "/create-statute-subtopics",
                        {
                            "column": "sub_chunk_text",
                            "database": db,
                            "destination_collection": "statute_subtopics",
                            "parse_prompt": "",
                            "source_collection": "statute_sub_chunks",
                            "source_query": source_query,
                            "subsection_column": "sub_chunk_text",
                        },
                    )
                    # 3. create-embeddings
                    _run_pipeline_step(
                        client,
                        "create-embeddings",
                        "/create-embeddings",
                        {
                            "index_collection_name": "statute_sub_embeddings",
                            "index_database_name": db,
                            "source_collection_name": "statute_sub_chunks",
                            "source_database_name": db,
                            "source_query": source_query,
                            "text_column": "sub_chunk_text",
                        },
                    )
                    # 4. create-vector-index
                    _run_pipeline_step(
                        client,
                        "create-vector-index",
                        "/create-vector-index",
                        {
                            "collection_name": "statute_sub_embeddings",
                            "database_name": db,
                            "filter_fields": ["jurisdiction", "document_id"],
                            "index_name": "vector_index",
                        },
                    )

                elif document_type == DOCUMENT_TYPE_POLICY:
                    # 1. create-policy-subsections
                    _run_pipeline_step(
                        client,
                        "create-policy-subsections",
                        "/create-policy-subsections",
                        {
                            "column": "chunk_text",
                            "database": db,
                            "destination_collection": "policy_sub_chunks",
                            "parse_prompt": "",
                            "source_collection": "policy_chunks",
                            "source_query": source_query,
                            "subsection_column": "sub_chunk_text",
                        },
                    )
                    # 2. create-embeddings
                    _run_pipeline_step(
                        client,
                        "create-embeddings",
                        "/create-embeddings",
                        {
                            "index_collection_name": "policy_sub_embeddings",
                            "index_database_name": db,
                            "source_collection_name": "policy_sub_chunks",
                            "source_database_name": db,
                            "source_query": source_query,
                            "text_column": "sub_chunk_text",
                        },
                    )
                    # 3. create-vector-index
                    _run_pipeline_step(
                        client,
                        "create-vector-index",
                        "/create-vector-index",
                        {
                            "collection_name": "policy_sub_embeddings",
                            "database_name": db,
                            "filter_fields": ["document_id"],
                            "index_name": "vector_index",
                        },
                    )
                else:
                    raise ValueError(f"Invalid document_type: {document_type}")

            result_dict = {
                "document_type": document_type,
                "source_query": source_query,
                "status": "completed",
            }
            job_storage.update_job_status(job_id, JOB_STATUS_COMPLETED, result=result_dict)
            return {
                "status": JOB_STATUS_COMPLETED,
                "result": result_dict,
                "completed_at": _iso(),
            }
        except Exception as e:
            logger.exception("Sub-vector-index job %s failed", job_id)
            error_msg = str(e)
            job_storage.update_job_status(job_id, JOB_STATUS_FAILED, error=error_msg)
            return {
                "status": JOB_STATUS_FAILED,
                "error": error_msg,
                "completed_at": _iso(),
            }

    builder = StateGraph(IndexJobState)
    builder.add_node("validate_and_start", validate_and_start)
    builder.add_node("run_pipeline", run_pipeline_node)
    builder.add_edge(START, "validate_and_start")
    builder.add_edge("validate_and_start", "run_pipeline")
    builder.add_edge("run_pipeline", END)
    return builder.compile()


def start_sub_vector_index_job(
    request_dict: Dict[str, Any],
    job_storage: IndexJobStorage,
    flask_app: Any,
) -> str:
    """
    Create job, start it in background, return job_id.
    The job runs in a separate thread with its own event loop.
    """
    job_id = job_storage.create_job(JOB_TYPE_SUB_VECTOR_INDEX, request_dict)
    graph = build_sub_vector_index_graph(flask_app, job_storage)

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            initial_state: IndexJobState = {
                "job_id": job_id,
                "job_type": JOB_TYPE_SUB_VECTOR_INDEX,
                "status": JOB_STATUS_PENDING,
                "request": request_dict,
            }
            loop.run_until_complete(graph.ainvoke(initial_state))
        except Exception as e:
            logger.exception("Background sub-vector-index job %s failed", job_id)
            job_storage.update_job_status(job_id, JOB_STATUS_FAILED, error=str(e))
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return job_id
