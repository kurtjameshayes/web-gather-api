"""Flask routes for policy statute compliance."""
from __future__ import annotations

import logging
import os

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from audit_logger import AuditLogger
from cache import SimpleLRUCache
from compliance_config import ComplianceConfig, load_config
from compliance_evaluator import ComplianceEvaluator
from compliance_service import ComplianceService, ServiceError
from compliance_storage import ComplianceStorage
from compliance_suite_schemas import (
    ApplicabilityRequest,
    DriftCheckRequest,
    GapAnalysisRequest,
    HealthScoreRequest,
    MultiJurisdictionalRequest,
)
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError
from db import get_embedding_model_name, set_application_embedding_model
from embedder import Embedder
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter
from redactor import Redactor
from schemas import PolicyStatuteComplianceRequest
from security import AuthorizationError, authorize_request
from segmenter import PolicySegmenter
from vector_retriever import VectorRetriever

logger = logging.getLogger("policy-compliance")

compliance_bp = Blueprint("compliance", __name__)

_service: ComplianceService | None = None
_suite_service: ComplianceSuiteService | None = None
_config: ComplianceConfig | None = None


def init_compliance(mongo_client) -> None:
    global _service, _suite_service, _config
    _config = load_config()

    # Resolve embedding model from web-gather for privacy-compliance; set app default for all vector queries.
    model_name = get_embedding_model_name(_config.compliance_database)
    if not model_name:
        model_name = _config.embedding_model_name
        logger.info(
            "No embedding model for %s in DB; using config: %s",
            _config.compliance_database,
            model_name,
        )
    else:
        logger.info(
            "Using embedding model from web-gather.embedding_model: %s",
            model_name,
        )
    set_application_embedding_model(model_name)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    embedding_cache = SimpleLRUCache(_config.cache_size, _config.cache_ttl_seconds)
    retrieval_cache = SimpleLRUCache(_config.cache_size, _config.cache_ttl_seconds)
    embedder = Embedder(model_name, embedding_cache)
    retriever = VectorRetriever(mongo_client, embedder, _config, retrieval_cache)
    llm_client = AnthropicLLMClient(api_key, _config)
    evaluator = ComplianceEvaluator(_config.evidence_score_threshold)
    segmenter = PolicySegmenter(_config.max_section_chars)
    redactor = Redactor()
    audit_logger = AuditLogger(mongo_client, _config)
    rate_limiter = RateLimiter(_config.rate_limit_per_minute)
    storage = ComplianceStorage(mongo_client, _config)

    _service = ComplianceService(
        mongo_client=mongo_client,
        config=_config,
        segmenter=segmenter,
        retriever=retriever,
        llm_client=llm_client,
        evaluator=evaluator,
        redactor=redactor,
        audit_logger=audit_logger,
        rate_limiter=rate_limiter,
    )
    _suite_service = ComplianceSuiteService(
        mongo_client=mongo_client,
        config=_config,
        retriever=retriever,
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
    )


def set_compliance_service(service: ComplianceService | None, config: ComplianceConfig | None = None) -> None:
    global _service, _config
    _service = service
    if config is not None:
        _config = config


def _get_service() -> ComplianceService:
    if _service is None:
        raise RuntimeError("Compliance service not initialized.")
    return _service


def _get_config() -> ComplianceConfig:
    if _config is None:
        raise RuntimeError("Compliance config not initialized.")
    return _config


@compliance_bp.post("/policy-statute-compliance")
async def policy_statute_compliance():
    logger.info("policy_statute_compliance received request: %s %s", request.method, request.path)
    logger.info("POST /policy-statute-compliance - Starting compliance check")
    payload = request.get_json(silent=True) or {}
    logger.info(
        "POST /policy-statute-compliance - Parameters: policy_collection=%s, policy_id=%s, jurisdiction=%s",
        payload.get("policy_collection"), payload.get("policy_id"), payload.get("jurisdiction"),
    )

    try:
        request_model = PolicyStatuteComplianceRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422

    try:
        config = _get_config()
        authorize_request(config, request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    try:
        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "compliance_routes.py:before_compare_policy", "message": "Calling compare_policy", "data": {"policy_id": request_model.policy_id, "jurisdiction": request_model.jurisdiction}, "hypothesisId": "H2"}) + "\n")
        except Exception:
            pass
        # #endregion
        result = await _get_service().compare_policy(request_model)
        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "compliance_routes.py:after_compare_policy", "message": "compare_policy returned", "data": {"sections_count": len(result.sections)}, "hypothesisId": "H2"}) + "\n")
        except Exception:
            pass
        # #endregion
        return jsonify(result.model_dump())
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as e:  # pragma: no cover - defensive fallback
        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "compliance_routes.py:exception", "message": "Unhandled error", "data": {"type": type(e).__name__, "str": str(e)}, "hypothesisId": "H2"}) + "\n")
        except Exception:
            pass
        # #endregion
        logger.exception("Unhandled error in policy_statute_compliance")
        return jsonify({"error": "Internal server error"}), 500


def _get_suite_service() -> ComplianceSuiteService:
    if _suite_service is None:
        raise RuntimeError("Compliance suite service not initialized.")
    return _suite_service


@compliance_bp.post("/applicability")
async def applicability():
    logger.info("POST /applicability - Jurisdiction inference")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = ApplicabilityRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        _get_config()
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().applicability(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in applicability")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.post("/gap-analysis")
async def gap_analysis():
    logger.info("POST /gap-analysis - Gap analysis")
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
        result = await _get_suite_service().gap_analysis(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in gap_analysis")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.post("/multi-jurisdictional")
async def multi_jurisdictional():
    logger.info("POST /multi-jurisdictional - Strictest common denominator")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = MultiJurisdictionalRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().multi_jurisdictional(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in multi_jurisdictional")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.post("/health-score")
async def health_score():
    logger.info("POST /health-score - Privacy health score")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = HealthScoreRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().health_score(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in health_score")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.post("/drift-check")
async def drift_check():
    logger.info("POST /drift-check - Regulatory drift alerts")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = DriftCheckRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().drift_check(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in drift_check")
        return jsonify({"error": "Internal server error"}), 500
