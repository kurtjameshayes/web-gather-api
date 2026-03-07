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
    AlertsListResponse,
    AlertListItem,
    ApplicabilityRequest,
    CitationsRequest,
    DriftCheckRequest,
    GapAnalysisRequest,
    HealthScoreRequest,
    MultiJurisdictionalRequest,
    ReportRequest,
    RiskAssessmentRequest,
    RunSummaryItem,
    RunsListResponse,
)
from compliance_job_service import ComplianceJobStorage, start_gap_analysis_job, start_health_score_job
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError
from gap_analysis_service_v3 import GapAnalysisServiceV3, GapAnalysisServiceV3Error
from gap_analysis_service_v4 import GapAnalysisServiceV4, GapAnalysisServiceV4Error
from db import ensure_privacy_compliance_indexes, get_embedding_model_name, set_application_embedding_model
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
_gap_analysis_v3_service: GapAnalysisServiceV3 | None = None
_gap_analysis_v4_service: GapAnalysisServiceV4 | None = None
_job_storage: ComplianceJobStorage | None = None
_config: ComplianceConfig | None = None


def init_compliance(mongo_client) -> None:
    global _service, _suite_service, _gap_analysis_v3_service, _gap_analysis_v4_service, _job_storage, _config
    _config = load_config()

    # Ensure MongoDB indexes on privacy-compliance collections (idempotent).
    ensure_privacy_compliance_indexes(mongo_client, _config.compliance_database)

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
    _gap_analysis_v4_service = GapAnalysisServiceV4(
        mongo_client=mongo_client,
        config=_config,
        llm_client=llm_client,
        rate_limiter=rate_limiter,
    )
    _suite_service = ComplianceSuiteService(
        mongo_client=mongo_client,
        config=_config,
        retriever=retriever,
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
        gap_analysis_v4_service=_gap_analysis_v4_service,
    )
    _gap_analysis_v3_service = GapAnalysisServiceV3(
        mongo_client=mongo_client,
        config=_config,
        retriever=retriever,
        llm_client=llm_client,
        storage=storage,
        rate_limiter=rate_limiter,
    )
    _job_storage = ComplianceJobStorage(mongo_client, _config)


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
        result = await _get_service().compare_policy(request_model)
        return jsonify(result.model_dump())
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled error in policy_statute_compliance")
        return jsonify({"error": "Internal server error"}), 500


def _get_suite_service() -> ComplianceSuiteService:
    if _suite_service is None:
        raise RuntimeError("Compliance suite service not initialized.")
    return _suite_service


def _get_gap_analysis_v3_service() -> GapAnalysisServiceV3:
    if _gap_analysis_v3_service is None:
        raise RuntimeError("Gap analysis v3 service not initialized.")
    return _gap_analysis_v3_service


def _get_gap_analysis_v4_service() -> GapAnalysisServiceV4:
    if _gap_analysis_v4_service is None:
        raise RuntimeError("Gap analysis v4 service not initialized.")
    return _gap_analysis_v4_service


def _get_job_storage() -> ComplianceJobStorage:
    if _job_storage is None:
        raise RuntimeError("Compliance job storage not initialized.")
    return _job_storage


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
        if request_model.run_async:
            # Start background job and return immediately once job has started.
            # Job creates compliance_results only when action completes (save_results=False).
            request_dict = request_model.model_dump(exclude={"run_async"})
            request_dict["save_results"] = False
            job_id = start_gap_analysis_job(
                request_dict=request_dict,
                job_storage=_get_job_storage(),
                run_gap_analysis_fn=_get_suite_service().gap_analysis,
                compliance_storage=_get_suite_service()._storage,
            )
            return jsonify({
                "job_id": job_id,
                "status": "pending",
                "message": "Gap analysis job started. Use GET /api/compliance/jobs/{job_id} to check status.",
            }), 202
        result = await _get_suite_service().gap_analysis(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception as e:
        logger.exception("Unhandled error in gap_analysis")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.get("/jobs/<job_id>")
async def get_job(job_id: str):
    """Get compliance job status and result (when completed)."""
    logger.info("GET /jobs/%s - Get job status", job_id)
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    job = _get_job_storage().get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


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
        if request_model.run_async:
            # Start background job and return immediately once job has started.
            # Job creates compliance_results only when action completes (save_results=False).
            request_dict = request_model.model_dump(exclude={"run_async"})
            request_dict["save_results"] = False
            job_id = start_health_score_job(
                request_dict=request_dict,
                job_storage=_get_job_storage(),
                run_health_score_fn=_get_suite_service().health_score,
                compliance_storage=_get_suite_service()._storage,
            )
            return jsonify({
                "job_id": job_id,
                "status": "pending",
                "message": "Health score job started. Use GET /api/compliance/jobs/{job_id} to check status.",
            }), 202
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


# ----- New spec endpoints (report, runs, citations, risk-assessment, alerts) -----


def _parse_iso8601(s: str | None) -> str | None:
    if not s or not s.strip():
        return None
    try:
        from datetime import datetime
        datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        return s.strip()
    except ValueError:
        return None


@compliance_bp.post("/report")
async def report():
    logger.info("POST /report - Generate compliance report")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = ReportRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().report(request_model)
        return jsonify(result)
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in report")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.get("/runs")
async def list_runs():
    logger.info("GET /runs - List compliance runs")
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    policy_document_id = request.args.get("policy_document_id") or None
    since_arg = request.args.get("since")
    until_arg = request.args.get("until")
    since = _parse_iso8601(since_arg) if since_arg is not None else None
    until = _parse_iso8601(until_arg) if until_arg is not None else None
    if since_arg is not None and since is None:
        return jsonify({"error": "Invalid since (expected ISO8601)"}), 400
    if until_arg is not None and until is None:
        return jsonify({"error": "Invalid until (expected ISO8601)"}), 400
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(0, min(200, limit))
    except ValueError:
        limit = 50
    try:
        offset = int(request.args.get("offset", 0))
        offset = max(0, offset)
    except ValueError:
        offset = 0
    types_param = request.args.get("types")
    types = [t.strip() for t in types_param.split(",") if t.strip()] if types_param else None
    if types_param and not types:
        types = None

    try:
        run_summaries, total = await _get_suite_service()._storage.list_runs(
            policy_document_id=policy_document_id,
            since=since,
            until=until,
            types=types,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.exception("Error listing runs")
        return jsonify({"error": str(e)}), 502

    runs = [RunSummaryItem(**s) for s in run_summaries]
    resp = RunsListResponse(runs=runs, total=total, limit=limit, offset=offset)
    return jsonify(resp.model_dump())


@compliance_bp.get("/runs/<run_id>")
async def get_run(run_id: str):
    logger.info("GET /runs/%s - Get run detail", run_id)
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    doc = await _get_suite_service()._storage.get_run_by_id(run_id)
    if not doc:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(doc)


@compliance_bp.post("/citations")
async def citations():
    logger.info("POST /citations - Statute-policy citation extraction")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = CitationsRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().citations(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in citations")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.post("/risk-assessment")
async def risk_assessment():
    logger.info("POST /risk-assessment - DPIA/PIA-style assessment")
    payload = request.get_json(silent=True) or {}
    try:
        request_model = RiskAssessmentRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation error", "details": exc.errors()}), 422
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        result = await _get_suite_service().risk_assessment(request_model)
        return jsonify(result.model_dump())
    except ComplianceSuiteServiceError as exc:
        return jsonify({"error": str(exc)}), exc.status_code
    except Exception:
        logger.exception("Unhandled error in risk_assessment")
        return jsonify({"error": "Internal server error"}), 500


@compliance_bp.get("/risk-assessment/templates")
async def risk_assessment_templates():
    logger.info("GET /risk-assessment/templates - List assessment templates")
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    result = await _get_suite_service().list_templates()
    return jsonify(result.model_dump())


@compliance_bp.get("/alerts")
async def list_alerts():
    logger.info("GET /alerts - List drift alerts")
    try:
        authorize_request(_get_config(), request)
    except AuthorizationError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    policy_document_id = request.args.get("policy_document_id") or None
    company_name = request.args.get("company_name") or None
    jurisdiction = request.args.get("jurisdiction") or None
    since = _parse_iso8601(request.args.get("since"))
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(0, min(200, limit))
    except ValueError:
        limit = 50
    try:
        offset = int(request.args.get("offset", 0))
        offset = max(0, offset)
    except ValueError:
        offset = 0

    try:
        alerts_docs, total = await _get_suite_service()._storage.list_alerts(
            policy_document_id=policy_document_id,
            company_name=company_name,
            jurisdiction=jurisdiction,
            since=since,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.exception("Error listing alerts")
        return jsonify({"error": str(e)}), 502

    alerts = [AlertListItem(**a) for a in alerts_docs]
    resp = AlertsListResponse(alerts=alerts, total=total, limit=limit, offset=offset)
    return jsonify(resp.model_dump())
