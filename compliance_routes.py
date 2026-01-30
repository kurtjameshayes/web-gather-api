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
_config: ComplianceConfig | None = None


def init_compliance(mongo_client) -> None:
    global _service, _config
    _config = load_config()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    embedding_cache = SimpleLRUCache(_config.cache_size, _config.cache_ttl_seconds)
    retrieval_cache = SimpleLRUCache(_config.cache_size, _config.cache_ttl_seconds)
    embedder = Embedder(_config.embedding_model_name, embedding_cache)
    retriever = VectorRetriever(mongo_client, embedder, _config, retrieval_cache)
    llm_client = AnthropicLLMClient(api_key, _config)
    evaluator = ComplianceEvaluator(_config.evidence_score_threshold)
    segmenter = PolicySegmenter(_config.max_section_chars)
    redactor = Redactor()
    audit_logger = AuditLogger(mongo_client, _config)
    rate_limiter = RateLimiter(_config.rate_limit_per_minute)

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
    logger.info("POST /policy-statute-compliance - Starting compliance check")
    payload = request.get_json(silent=True) or {}
    logger.info("POST /policy-statute-compliance - Parameters: database=%s, policy_collection=%s, "
                "policy_id=%s, text=%s, jurisdiction=%s, statute_corpus_id=%s, top_k_statutes=%s, "
                "confidence_thresholds=%s, options=%s",
                payload.get("database"), payload.get("policy_collection"), payload.get("policy_id"),
                "provided" if payload.get("text") else None, payload.get("jurisdiction"),
                payload.get("statute_corpus_id"), payload.get("top_k_statutes"),
                payload.get("confidence_thresholds"), payload.get("options"))

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
        result = await _get_service().compare_policy(request_model)
        return jsonify(result.model_dump())
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled error in policy_statute_compliance")
        return jsonify({"error": "Internal server error"}), 500
