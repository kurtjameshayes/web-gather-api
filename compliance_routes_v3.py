"""Flask routes for compliance API v3 (gap analysis per GapAnalysisProcessDesign.md)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from compliance_job_service import start_gap_analysis_job
from compliance_routes import _get_config, _get_gap_analysis_v3_service, _get_job_storage, _get_suite_service
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v3 import GapAnalysisServiceV3Error
from security import AuthorizationError, authorize_request

logger = logging.getLogger("policy-compliance")

compliance_v3_bp = Blueprint("compliance_v3", __name__)

COMPLIANCE_V3_API_PREFIX = "/api/v3/compliance"


def _start_v3_gap_analysis_job(request_dict: dict) -> str:
    """Start v3 gap analysis in background. Job creates compliance_results only when action completes."""
    async def run_v3(req):
        return await _get_gap_analysis_v3_service().run(req)
    request_dict = dict(request_dict)
    request_dict["save_results"] = False
    return start_gap_analysis_job(
        request_dict,
        _get_job_storage(),
        run_v3,
        compliance_storage=_get_suite_service()._storage,
    )


@compliance_v3_bp.post("/gap-analysis")
async def gap_analysis_v3():
    """Gap analysis v3: statute→policy vector search, top-k matches, score threshold, LLM per pair, citation binding."""
    logger.info("POST /gap-analysis (v3) - Gap analysis per GapAnalysisProcessDesign.md")
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
            job_id = _start_v3_gap_analysis_job(request_dict)
            return jsonify({
                "job_id": job_id,
                "status": "pending",
                "message": "Gap analysis v3 job started. Use GET /api/compliance/jobs/{job_id} to check status.",
            }), 202
        result = await _get_gap_analysis_v3_service().run(request_model)
        return jsonify(result.model_dump())
    except GapAnalysisServiceV3Error as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as exc:
        if hasattr(exc, "status_code"):
            return jsonify({"error": str(exc)}), exc.status_code
        logger.exception("Unhandled error in gap_analysis_v3")
        return jsonify({"error": "Internal server error"}), 500
