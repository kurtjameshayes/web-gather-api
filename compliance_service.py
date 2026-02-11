"""Service orchestration for policy statute compliance checks."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from audit_logger import AuditLogger
from compliance_config import ComplianceConfig
from compliance_evaluator import ComplianceEvaluator
from compliance_utils import normalize_jurisdiction, validate_collection_name
from llm_client import LLMClient
from rate_limiter import RateLimiter
from redactor import Redactor
from schemas import (
    PolicyStatuteComplianceRequest,
    PolicyStatuteComplianceResponse,
    PolicySectionResult,
    SummaryCounts,
    SummaryResult,
)
from segmenter import PolicySegmenter
from vector_retriever import StatuteCandidate, VectorRetriever

logger = logging.getLogger("policy-compliance")


async def _run_in_thread(func):
    """Run a sync function in a thread (Python 3.8 compat: no asyncio.to_thread)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class ComplianceService:
    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        segmenter: PolicySegmenter,
        retriever: VectorRetriever,
        llm_client: LLMClient,
        evaluator: ComplianceEvaluator,
        redactor: Redactor,
        audit_logger: AuditLogger,
        rate_limiter: RateLimiter,
    ) -> None:
        self._mongo_client = mongo_client
        self._config = config
        self._segmenter = segmenter
        self._retriever = retriever
        self._llm_client = llm_client
        self._evaluator = evaluator
        self._redactor = redactor
        self._audit_logger = audit_logger
        self._rate_limiter = rate_limiter
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the current event loop (avoids 'Future attached to a different loop')."""
        loop = asyncio.get_event_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore_loop = loop
            self._semaphore = asyncio.Semaphore(self._config.llm_concurrency)
        return self._semaphore

    async def compare_policy(
        self, payload: PolicyStatuteComplianceRequest
    ) -> PolicyStatuteComplianceResponse:
        if not await self._rate_limiter.allow():
            raise ServiceError("Rate limit exceeded", status_code=429)

        warnings: List[str] = []
        jurisdiction = normalize_jurisdiction(payload.jurisdiction)
        database = self._config.compliance_database
        retrieval_database = (self._config.statute_database or "").strip() or database
        top_k = self._config.top_k_statutes
        thresholds = self._config.confidence_thresholds

        policy_text, existing_sections = await self._load_policy(
            database, payload.policy_collection, payload.policy_id, None
        )

        if not policy_text:
            raise ServiceError("Policy text not found or empty.", status_code=400)

        if self._config.enable_redaction:
            redaction_result = self._redactor.redact(policy_text)
            policy_text = redaction_result.redacted_text

        sections = self._segmenter.segment(policy_text, existing_sections)
        if not sections:
            raise ServiceError("No policy sections detected.", status_code=422)

        tasks = [
            self._process_section(
                section.section_id,
                section.section_text,
                database,
                retrieval_database,
                jurisdiction,
                None,
                top_k,
                thresholds,
                True,
                self._config.enable_redaction,
            )
            for section in sections
        ]

        results = await asyncio.gather(*tasks)
        response_sections: List[PolicySectionResult] = []
        for result in results:
            warnings.extend(result["warnings"])
            response_sections.append(
                PolicySectionResult(
                    section_id=result["section_id"],
                    section_text=result["section_text"],
                    applied_statutes=result["applied_statutes"],
                    compliance=result["compliance"],
                    confidence=result["confidence"],
                    rationale=result["rationale"],
                    remediation_suggestions=result["remediation_suggestions"],
                    retrieval_trace=result["retrieval_trace"],
                )
            )

        summary = self._build_summary(response_sections)

        await self._audit_logger.log(
            database,
            payload.policy_id,
            jurisdiction,
            policy_text,
            [section.model_dump() for section in response_sections],
            summary.model_dump(),
        )

        return PolicyStatuteComplianceResponse(
            policy_id=payload.policy_id,
            jurisdiction=jurisdiction,
            sections=response_sections,
            summary=summary,
            warnings=warnings,
        )

    async def _load_policy(
        self,
        database: str,
        policy_collection: str,
        policy_id: Optional[str],
        inline_text: Optional[str],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not validate_collection_name(database) or not validate_collection_name(policy_collection):
            raise ServiceError("Invalid database or collection name.", status_code=400)
        if inline_text:
            return inline_text.strip(), None
        if not policy_id:
            return "", None

        collection = self._mongo_client[database][policy_collection]
        doc_id_field = self._config.policy_document_id_field
        chunk_index_field = self._config.policy_chunk_index_field
        chunk_text_field = self._config.policy_chunk_text_field
        chunk_header_field = self._config.policy_chunk_header_field

        def run_find_chunks() -> List[Dict[str, Any]]:
            return list(
                collection.find({doc_id_field: policy_id}).sort(chunk_index_field, 1)
            )

        chunks = await _run_in_thread(run_find_chunks)
        if not chunks:
            return "", None

        parts: List[str] = []
        for chunk in chunks:
            header = (chunk.get(chunk_header_field) or "").strip()
            text_part = (chunk.get(chunk_text_field) or "").strip()
            if header:
                parts.append(header)
            if text_part:
                parts.append(text_part)
        return "\n\n".join(parts).strip(), None

    async def _process_section(
        self,
        section_id: str,
        section_text: str,
        database: str,
        retrieval_database: str,
        jurisdiction: str,
        statute_corpus_id: Optional[str],
        top_k: int,
        thresholds: Any,
        explainability: bool,
        redact_pii: bool,
    ) -> Dict[str, Any]:
        async with self._get_semaphore():
            if redact_pii and self._config.enable_redaction:
                section_text = self._redactor.redact(section_text).redacted_text

            candidates = await self._retriever.retrieve(
                retrieval_database,
                section_text,
                jurisdiction,
                statute_corpus_id,
                top_k,
            )

            if not candidates:
                retrieval_trace = (
                    [f"0 candidates (jurisdiction={jurisdiction})"]
                    if explainability
                    else []
                )
                return {
                    "section_id": section_id,
                    "section_text": section_text,
                    "applied_statutes": [],
                    "compliance": "neither",
                    "confidence": 0.0,
                    "rationale": "No statute candidates retrieved for this section and jurisdiction.",
                    "remediation_suggestions": [
                        "Verify statute vector index and jurisdiction filter (see compliance README)."
                    ],
                    "retrieval_trace": retrieval_trace,
                    "warnings": [],
                }

            llm_response = await self._llm_client.compare_section(
                section_id, section_text, candidates
            )

            evaluated = self._evaluator.evaluate(
                section_id=section_id,
                section_text=section_text,
                candidates=candidates,
                llm_response_text=llm_response,
                thresholds=thresholds,
                explainability=explainability,
            )

            return {
                "section_id": evaluated.section_id,
                "section_text": evaluated.section_text,
                "applied_statutes": evaluated.applied_statutes,
                "compliance": evaluated.compliance,
                "confidence": evaluated.confidence,
                "rationale": evaluated.rationale,
                "remediation_suggestions": evaluated.remediation_suggestions,
                "retrieval_trace": evaluated.retrieval_trace,
                "warnings": evaluated.warnings,
            }

    def _build_summary(self, sections: List[PolicySectionResult]) -> SummaryResult:
        compliant = sum(1 for section in sections if section.compliance == "compliant")
        non_compliant = sum(
            1 for section in sections if section.compliance == "non_compliant"
        )
        neither = sum(1 for section in sections if section.compliance == "neither")

        overall = "unknown"
        total = len(sections)
        if total == 0:
            overall = "unknown"
        elif compliant == total:
            overall = "compliant"
        elif non_compliant > 0 and non_compliant > compliant:
            overall = "non_compliant"
        else:
            overall = "mixed"

        return SummaryResult(
            overall_compliance=overall,
            counts=SummaryCounts(
                compliant=compliant, non_compliant=non_compliant, neither=neither
            ),
        )
