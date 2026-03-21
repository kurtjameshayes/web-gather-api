"""Flask routes for compliance API v4 (category-mapping-driven gap analysis)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from compliance_job_service import start_gap_analysis_job
from compliance_routes import _get_config, _get_critic_service, _get_gap_analysis_v4_service, _get_job_storage, _get_suite_service
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4Error
from security import AuthorizationError, authorize_request

logger = logging.getLogger("policy-compliance")

compliance_v4_bp = Blueprint("compliance_v4", __name__)

COMPLIANCE_V4_API_PREFIX = "/api/v4/compliance"


def _start_v4_gap_analysis_job(request_dict: dict) -> str:
    """Start v4 gap analysis in background. Job creates compliance_results only when action completes."""
    svc = _get_gap_analysis_v4_service()
    async def run_v4(req):
        return await svc.run(req)
    request_dict = dict(request_dict)
    request_dict["save_results"] = False
    return start_gap_analysis_job(
        request_dict,
        _get_job_storage(),
        run_v4,
        compliance_storage=_get_suite_service()._storage,
    )


@compliance_v4_bp.post("/gap-analysis")
async def gap_analysis_v4():
    """Gap analysis v4: category-mapping-driven, statute_sub_topic_embeddings + policy_legal_embeddings."""
    logger.info("POST /gap-analysis (v4) - Category-mapping-driven gap analysis")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = GapAnalysisRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        if request_model.run_async:
            request_dict = request_model.model_dump(exclude={"run_async"})
            job_id = _start_v4_gap_analysis_job(request_dict)
            return jsonify({
                "job_id": job_id,
                "status": "pending",
                "message": "Gap analysis v4 job started. Use GET /api/compliance/jobs/{job_id} to check status.",
            }), 202
        result = await _get_gap_analysis_v4_service().run(request_model)
        return jsonify(result.model_dump())
    except GapAnalysisServiceV4Error as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:
        if hasattr(exc, "status_code"):
            return jsonify({"error": str(exc)}), exc.status_code
        logger.exception("Unhandled error in gap_analysis_v4")
        return jsonify({"error": "Internal server error"}), 500


@compliance_v4_bp.get("/adaptive-feedback")
def list_adaptive_feedback():
    """List adaptive feedback history for a policy."""
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    policy_document_id = request.args.get("policy_document_id")
    if not policy_document_id:
        return jsonify({"error": "policy_document_id query parameter is required"}), 400

    active_only = request.args.get("active_only", "false").lower() in ("true", "1", "yes")
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 200))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = int(request.args.get("offset", 0))
        offset = max(0, offset)
    except (TypeError, ValueError):
        offset = 0

    critic = _get_critic_service()
    docs = critic.get_feedback_history(
        policy_document_id=policy_document_id,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )
    return jsonify({"feedback": docs, "limit": limit, "offset": offset})


@compliance_v4_bp.get("/adaptive-feedback/log")
def list_adaptive_feedback_log():
    """List which feedback was consumed by gap analysis runs."""
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    run_id = request.args.get("run_id")
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 200))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = int(request.args.get("offset", 0))
        offset = max(0, offset)
    except (TypeError, ValueError):
        offset = 0

    critic = _get_critic_service()
    docs = critic.get_feedback_log(run_id=run_id, limit=limit, offset=offset)
    return jsonify({"log": docs, "limit": limit, "offset": offset})
