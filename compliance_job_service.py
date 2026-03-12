"""Background job service for compliance features using LangGraph and MongoDB job storage."""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict

from compliance_config import ComplianceConfig
from compliance_suite_schemas import GapAnalysisRequest, HealthScoreRequest
from compliance_utils import utc_now

logger = logging.getLogger("policy-compliance")

# Job status values
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

# Job types
JOB_TYPE_GAP_ANALYSIS = "gap_analysis"
JOB_TYPE_HEALTH_SCORE = "health_score"
JOB_TYPE_DRIFT_CHECK = "drift_check"


def _iso(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


class ComplianceJobState(TypedDict, total=False):
    """State for LangGraph compliance job workflow."""

    job_id: str
    job_type: str
    status: str
    request: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class ComplianceJobStorage:
    """MongoDB storage for compliance background jobs."""

    def __init__(self, mongo_client: Any, config: ComplianceConfig) -> None:
        self._client = mongo_client
        self._config = config
        self._coll_name = config.compliance_jobs_collection
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


def build_gap_analysis_graph(
    run_gap_analysis_fn,
    job_storage: ComplianceJobStorage,
    compliance_storage: Optional[Any] = None,
) -> Any:
    """Build LangGraph StateGraph for gap analysis job. Falls back to compliance_graph if langgraph not installed (Python 3.8)."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        from compliance_graph import END, START, StateGraph

    def validate_and_start(state: ComplianceJobState) -> ComplianceJobState:
        """Validate request and mark job as running."""
        job_id = state["job_id"]
        job_storage.update_job_status(job_id, JOB_STATUS_RUNNING)
        return {
            "status": JOB_STATUS_RUNNING,
            "started_at": _iso(),
        }

    async def run_gap_analysis_node(state: ComplianceJobState) -> ComplianceJobState:
        """Execute gap analysis; write compliance_results only when action completes successfully."""
        job_id = state["job_id"]
        request_dict = state.get("request") or {}
        try:
            req = GapAnalysisRequest.model_validate(request_dict)
            result = await run_gap_analysis_fn(req)
            result_dict = result.model_dump() if hasattr(result, "model_dump") else result

            # Write compliance_results and run_log only after action completes successfully
            if compliance_storage:
                try:
                    run_types = result_dict.get("run_types") or ["gap"]
                    res_doc = {
                        "policy_document_id": result_dict.get("policy_document_id"),
                        "company_name": result_dict.get("company_name"),
                        "applicable_jurisdictions": result_dict.get("applicable_jurisdictions", []),
                        "gaps": result_dict.get("gaps", []),
                        "summary": result_dict.get("summary", {}),
                        "analyzed_at": result_dict.get("analyzed_at"),
                        "run_types": run_types,
                    }
                    if result_dict.get("run_type"):
                        res_doc["run_type"] = result_dict["run_type"]
                    if result_dict.get("version"):
                        res_doc["version"] = result_dict["version"]
                    if result_dict.get("retrieval_metadata"):
                        res_doc["retrieval_metadata"] = result_dict["retrieval_metadata"]
                    await compliance_storage.write_compliance_result(res_doc)
                    log_doc = {
                        "policy_document_id": result_dict.get("policy_document_id"),
                        "applicable_jurisdictions": result_dict.get("applicable_jurisdictions", []),
                        "run_timestamp": result_dict.get("analyzed_at"),
                        "statute_chunk_ids_used": result_dict.get("statute_chunk_ids_used") or [],
                    }
                    if result_dict.get("run_type"):
                        log_doc["run_type"] = result_dict["run_type"]
                    if result_dict.get("summary"):
                        log_doc["summary"] = result_dict["summary"]
                    await compliance_storage.write_compliance_run_log(log_doc)
                except Exception as e:
                    logger.warning("Failed to write compliance result/run_log for job %s: %s", job_id, e)

            job_storage.update_job_status(
                job_id,
                JOB_STATUS_COMPLETED,
                result=result_dict,
            )
            return {
                "status": JOB_STATUS_COMPLETED,
                "result": result_dict,
                "completed_at": _iso(),
            }
        except Exception as e:
            logger.exception("Gap analysis job %s failed", job_id)
            error_msg = str(e)
            job_storage.update_job_status(
                job_id,
                JOB_STATUS_FAILED,
                error=error_msg,
            )
            return {
                "status": JOB_STATUS_FAILED,
                "error": error_msg,
                "completed_at": _iso(),
            }

    builder = StateGraph(ComplianceJobState)
    builder.add_node("validate_and_start", validate_and_start)
    builder.add_node("run_gap_analysis", run_gap_analysis_node)
    builder.add_edge(START, "validate_and_start")
    builder.add_edge("validate_and_start", "run_gap_analysis")
    builder.add_edge("run_gap_analysis", END)
    return builder.compile()


def start_gap_analysis_job(
    request_dict: Dict[str, Any],
    job_storage: ComplianceJobStorage,
    run_gap_analysis_fn,
    compliance_storage: Optional[Any] = None,
) -> str:
    """
    Create job, start it in background, return job_id.
    The job runs in a separate thread with its own event loop.
    """
    job_id = job_storage.create_job(JOB_TYPE_GAP_ANALYSIS, request_dict)
    graph = build_gap_analysis_graph(run_gap_analysis_fn, job_storage, compliance_storage)

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            initial_state: ComplianceJobState = {
                "job_id": job_id,
                "job_type": JOB_TYPE_GAP_ANALYSIS,
                "status": JOB_STATUS_PENDING,
                "request": request_dict,
            }
            loop.run_until_complete(graph.ainvoke(initial_state))
        except Exception as e:
            logger.exception("Background gap analysis job %s failed", job_id)
            job_storage.update_job_status(job_id, JOB_STATUS_FAILED, error=str(e))
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return job_id


def build_health_score_graph(
    run_health_score_fn,
    job_storage: ComplianceJobStorage,
    compliance_storage: Optional[Any] = None,
) -> Any:
    """Build LangGraph StateGraph for health score job. Falls back to compliance_graph if langgraph not installed (Python 3.8)."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        from compliance_graph import END, START, StateGraph

    def validate_and_start(state: ComplianceJobState) -> ComplianceJobState:
        """Validate request and mark job as running."""
        job_id = state["job_id"]
        job_storage.update_job_status(job_id, JOB_STATUS_RUNNING)
        return {
            "status": JOB_STATUS_RUNNING,
            "started_at": _iso(),
        }

    async def run_health_score_node(state: ComplianceJobState) -> ComplianceJobState:
        """Execute health score; write compliance_results only when action completes successfully."""
        job_id = state["job_id"]
        request_dict = state.get("request") or {}
        try:
            req = HealthScoreRequest.model_validate(request_dict)
            result = await run_health_score_fn(req)
            result_dict = result.model_dump() if hasattr(result, "model_dump") else result

            # Write compliance_results only after action completes successfully
            if compliance_storage:
                try:
                    res_doc = {
                        "policy_document_id": result_dict.get("policy_document_id"),
                        "company_name": result_dict.get("company_name"),
                        "applicable_jurisdictions": result_dict.get("applicable_jurisdictions", []),
                        "privacy_health_score": result_dict.get("privacy_health_score"),
                        "score_assessment": result_dict.get("score_assessment"),
                        "score_breakdown": result_dict.get("score_breakdown", {}),
                        "components": result_dict.get("components", {}),
                        "analyzed_at": result_dict.get("analyzed_at"),
                        "run_types": ["health_score"],
                    }
                    await compliance_storage.write_compliance_result(res_doc)
                except Exception as e:
                    logger.warning("Failed to write compliance result for job %s: %s", job_id, e)

            job_storage.update_job_status(
                job_id,
                JOB_STATUS_COMPLETED,
                result=result_dict,
            )
            return {
                "status": JOB_STATUS_COMPLETED,
                "result": result_dict,
                "completed_at": _iso(),
            }
        except Exception as e:
            logger.exception("Health score job %s failed", job_id)
            error_msg = str(e)
            job_storage.update_job_status(
                job_id,
                JOB_STATUS_FAILED,
                error=error_msg,
            )
            return {
                "status": JOB_STATUS_FAILED,
                "error": error_msg,
                "completed_at": _iso(),
            }

    builder = StateGraph(ComplianceJobState)
    builder.add_node("validate_and_start", validate_and_start)
    builder.add_node("run_health_score", run_health_score_node)
    builder.add_edge(START, "validate_and_start")
    builder.add_edge("validate_and_start", "run_health_score")
    builder.add_edge("run_health_score", END)
    return builder.compile()


def start_health_score_job(
    request_dict: Dict[str, Any],
    job_storage: ComplianceJobStorage,
    run_health_score_fn,
    compliance_storage: Optional[Any] = None,
) -> str:
    """
    Create job, start it in background, return job_id.
    The job runs in a separate thread with its own event loop.
    """
    job_id = job_storage.create_job(JOB_TYPE_HEALTH_SCORE, request_dict)
    graph = build_health_score_graph(run_health_score_fn, job_storage, compliance_storage)

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            initial_state: ComplianceJobState = {
                "job_id": job_id,
                "job_type": JOB_TYPE_HEALTH_SCORE,
                "status": JOB_STATUS_PENDING,
                "request": request_dict,
            }
            loop.run_until_complete(graph.ainvoke(initial_state))
        except Exception as e:
            logger.exception("Background health score job %s failed", job_id)
            job_storage.update_job_status(job_id, JOB_STATUS_FAILED, error=str(e))
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return job_id
