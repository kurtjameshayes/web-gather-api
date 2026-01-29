"""FastAPI application for policy statute compliance."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request
from pymongo import MongoClient

from audit_logger import AuditLogger
from cache import SimpleLRUCache
from compliance_config import ComplianceConfig, load_config
from compliance_evaluator import ComplianceEvaluator
from compliance_service import ComplianceService, ServiceError
from embedder import Embedder
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter
from redactor import Redactor
from schemas import PolicyStatuteComplianceRequest, PolicyStatuteComplianceResponse
from security import authorize_request
from segmenter import PolicySegmenter
from vector_retriever import VectorRetriever

logger = logging.getLogger("policy-compliance")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Policy Statute Compliance API",
    version="1.0.0",
    description="Compare privacy policy sections against statutes for compliance.",
)


@lru_cache()
def get_config() -> ComplianceConfig:
    return load_config()


@lru_cache()
def get_mongo_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    return MongoClient(uri)


_service: ComplianceService | None = None


def get_service(
    config: ComplianceConfig = Depends(get_config),
    mongo_client: MongoClient = Depends(get_mongo_client),
) -> ComplianceService:
    global _service
    if _service is None:
        embedding_cache = SimpleLRUCache(config.cache_size, config.cache_ttl_seconds)
        retrieval_cache = SimpleLRUCache(config.cache_size, config.cache_ttl_seconds)
        embedder = Embedder(config.embedding_model_name, embedding_cache)
        retriever = VectorRetriever(mongo_client, embedder, config, retrieval_cache)
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        llm_client = AnthropicLLMClient(api_key, config)
        evaluator = ComplianceEvaluator(config.evidence_score_threshold)
        segmenter = PolicySegmenter(config.max_section_chars)
        redactor = Redactor()
        audit_logger = AuditLogger(mongo_client, config)
        rate_limiter = RateLimiter(config.rate_limit_per_minute)
        _service = ComplianceService(
            mongo_client,
            config,
            segmenter,
            retriever,
            llm_client,
            evaluator,
            redactor,
            audit_logger,
            rate_limiter,
        )
    return _service


@app.post(
    "/policy-statute-compliance",
    response_model=PolicyStatuteComplianceResponse,
    response_model_exclude_none=True,
)
@app.post(
    "/api/v1/compare-policy",
    response_model=PolicyStatuteComplianceResponse,
    response_model_exclude_none=True,
)
async def compare_policy(
    payload: PolicyStatuteComplianceRequest,
    request: Request,
    service: ComplianceService = Depends(get_service),
    config: ComplianceConfig = Depends(get_config),
) -> PolicyStatuteComplianceResponse:
    authorize_request(config, request)
    try:
        return await service.compare_policy(payload)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unhandled error in compare_policy")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
