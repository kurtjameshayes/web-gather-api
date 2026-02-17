"""Flask routes for compliance API v2 (chunk-level gap analysis)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from compliance_suite_schemas import GapAnalysisRequest
from compliance_routes import _get_config, _get_suite_service
from security import AuthorizationError, authorize_request

logger = logging.getLogger("policy-compliance")

compliance_v2_bp = Blueprint("compliance_v2", __name__)


@compliance_v2_bp.post("/gap-analysis")
async def gap_analysis_v2():
    """Chunk-level gap analysis: statute_embeddings vs policy_embeddings. Uses YAML-configurable prompt."""
    logger.info("POST /gap-analysis (v2) - Chunk-level gap analysis")
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
        result = await _get_suite_service().gap_analysis_v2(request_model)
        return jsonify(result.model_dump())
    except Exception as exc:
        if hasattr(exc, "status_code"):
            return jsonify({"error": str(exc)}), exc.status_code
        logger.exception("Unhandled error in gap_analysis_v2")
        return jsonify({"error": "Internal server error"}), 500
