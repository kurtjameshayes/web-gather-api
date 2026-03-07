"""Service layer for the compliance suite: applicability, gap analysis, health score, multi-jurisdictional, drift."""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId
from bson.errors import InvalidId

from compliance_config import ComplianceConfig
from compliance_suite_schemas import (
    ApplicabilityResponse,
    CitationItem,
    RetrievalMetadata,
    CitationsRequest,
    CitationsResponse,
    CitationsSummary,
    ConflictBetweenJurisdictionsItem,
    DriftAlertItem,
    DriftCheckResponse,
    DriftGapItem,
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreResponse,
    MultiJurisdictionalResponse,
    ReportRequest,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    StrictestDenominatorItem,
    TemplateItem,
    TemplatesResponse,
)
from compliance_storage import ComplianceStorage
from compliance_utils import truncate_at_sentence, utc_now
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter
from vector_retriever import ChunkPair, StatuteCandidate, SubchunkPair, VectorRetriever


async def _run_in_thread(func, *args, **kwargs):
    """Run sync function in a thread (Python 3.8 compat: asyncio.to_thread added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# Re-export for typing
from compliance_suite_schemas import (
    ApplicabilityRequest,
    DriftCheckRequest,
    GapAnalysisRequest,
    HealthScoreRequest,
    MultiJurisdictionalRequest,
)


def _iso(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


class ComplianceSuiteServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _citation_binding(policy_quote: Optional[str], policy_text: str) -> bool:
    """Return True if policy_quote is a substring of policy_text (normalize whitespace)."""
    if not policy_quote or not policy_text:
        return False
    q = re.sub(r"\s+", " ", policy_quote.strip())
    t = re.sub(r"\s+", " ", policy_text)
    return q in t if q else False


def _looks_like_section_citation(text: Optional[str]) -> bool:
    """Return True if text looks like a section citation (e.g. 1798.105(a), § 1798.105)."""
    if not text or not text.strip():
        return False
    t = text.strip()
    # Match patterns like 1798.105, 1798.105(a), § 1798.105
    return bool(re.search(r"\d+\.\d+(?:\([a-z]\))?", t))


def _section_value(section_id: str, chunk_header_text: str) -> Optional[str]:
    """Prefer chunk_header_text when it looks like a section citation; else section_id; else chunk_header_text."""
    if chunk_header_text and _looks_like_section_citation(chunk_header_text):
        return chunk_header_text.strip()
    return (section_id or chunk_header_text or "").strip() or None


class ComplianceSuiteService:
    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        retriever: VectorRetriever,
        llm_client: AnthropicLLMClient,
        storage: ComplianceStorage,
        rate_limiter: RateLimiter,
        gap_analysis_v4_service: Any = None,
    ) -> None:
        self._mongo_client = mongo_client
        self._config = config
        self._retriever = retriever
        self._llm_client = llm_client
        self._storage = storage
        self._rate_limiter = rate_limiter
        self._gap_analysis_v4_service = gap_analysis_v4_service
        self._semaphore = asyncio.Semaphore(config.llm_concurrency)
        self._database = config.compliance_database
        self._retrieval_database = (config.statute_database or "").strip() or config.compliance_database

    async def _load_policy_text(
        self,
        policy_document_id: str,
        database: Optional[str] = None,
        policy_collection: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Load policy full text and optional company_name. Returns (text, company_name).

        Queries by document_id field first (policy_document_id_field from config),
        then falls back to _id for backward compatibility.
        """
        db_name = database or self._database
        coll_name = policy_collection or self._config.policies_collection
        doc_id_field = self._config.policy_document_id_field

        def find():
            coll = self._mongo_client[db_name][coll_name]
            doc = coll.find_one({doc_id_field: policy_document_id})
            if not doc:
                try:
                    doc = coll.find_one({"_id": ObjectId(policy_document_id)})
                except (InvalidId, TypeError):
                    doc = coll.find_one({"_id": policy_document_id})
            return doc

        doc = await _run_in_thread(find)
        if not doc:
            return "", None
        text = (doc.get("text") or "").strip()
        if not text and doc.get("policy_chunks"):
            chunks = doc["policy_chunks"]
            if isinstance(chunks, list):
                parts = []
                for c in chunks:
                    if isinstance(c, dict) and c.get("chunk_text"):
                        parts.append(c["chunk_text"].strip())
                text = "\n\n".join(parts).strip()
        company_name = doc.get("company_name") if isinstance(doc.get("company_name"), str) else None
        return text, company_name

    async def applicability(self, req: ApplicabilityRequest) -> ApplicabilityResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)
        if req.text:
            policy_text = req.text.strip()
        else:
            policy_text, _ = await self._load_policy_text(
                req.policy_document_id or "",
                database=req.database,
                policy_collection=req.policy_collection,
            )
        if not policy_text:
            raise ComplianceSuiteServiceError("Policy text not found or empty.", status_code=400)
        out = await self._llm_client.applicability(policy_text)
        return ApplicabilityResponse(
            applicable_jurisdictions=out.get("applicable_jurisdictions", []) or [],
            confidence=out.get("confidence"),
        )

    async def gap_analysis(self, req: GapAnalysisRequest) -> GapAnalysisResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        if not policy_document_id:
            raise ComplianceSuiteServiceError("policy_document_id is required.", status_code=400)

        _, company_name = await self._load_policy_text(
            policy_document_id,
            database=req.database,
            policy_collection=req.policy_collection,
        )

        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        if not jurisdictions:
            appl = await self.applicability(
                ApplicabilityRequest(policy_document_id=policy_document_id, database=req.database, policy_collection=req.policy_collection)
            )
            jurisdictions = appl.applicable_jurisdictions or self._config.default_jurisdictions

        retrieve_result = await self._retriever.retrieve_policy_subchunks_for_statute_subchunks(
            database=self._retrieval_database,
            policy_document_id=policy_document_id,
            applicable_jurisdictions=jurisdictions,
            statute_document_id=req.statute_document_id,
            top_k_per_statute=1,
        )
        pairs = retrieve_result.pairs
        statute_subchunks_considered = retrieve_result.statute_subchunks_considered
        pairs_before_slice = len(pairs)
        if req.num_rows is not None:
            pairs = pairs[: req.num_rows]

        if not pairs:
            doc_id_field = self._config.policy_document_id_field
            pol_coll = self._mongo_client[self._retrieval_database][
                self._config.policy_sub_embeddings_collection
            ]

            def _count_policy_subchunks():
                return pol_coll.count_documents({doc_id_field: policy_document_id})

            n = await _run_in_thread(_count_policy_subchunks)
            if n == 0:
                raise ComplianceSuiteServiceError(
                    "Policy not indexed for gap analysis. Index this policy into "
                    f"{self._config.policy_sub_embeddings_collection} (run split-paragraph-chunks "
                    "then index-by-embedding) before running gap analysis.",
                    status_code=400,
                )

        gaps: List[GapItem] = []
        seen: Set[Tuple[str, str]] = set()
        statute_chunk_ids_used: List[str] = []

        stat_sub_field = self._config.statute_subchunk_text_field
        stat_chunk_field = self._config.statute_chunk_text_field
        pol_sub_field = self._config.policy_subchunk_text_field
        pol_chunk_field = self._config.policy_chunk_text_field
        jur_field = self._config.statute_jurisdiction_field

        for pair in pairs:
            stat_doc = pair.statute_doc
            pol_doc = pair.policy_doc
            statute_subchunk = (stat_doc.get(stat_sub_field) or "").strip()
            statute_chunk = (stat_doc.get(stat_chunk_field) or "").strip()
            policy_subchunk = (pol_doc.get(pol_sub_field) or "").strip()
            policy_chunk = (pol_doc.get(pol_chunk_field) or "").strip()

            subchunk_id = stat_doc.get("subchunk_id") or ""
            statute_ref = str(stat_doc.get("document_id") or stat_doc.get("source_id") or "")
            jurisdiction = str(stat_doc.get(jur_field) or "")

            key = (statute_ref, subchunk_id)
            if key in seen:
                continue
            seen.add(key)
            statute_chunk_ids_used.append(f"{statute_ref}:{subchunk_id}")

            async with self._semaphore:
                result = await self._llm_client.gap_check_subchunks(
                    statute_subchunk_text=statute_subchunk,
                    statute_chunk_text=statute_chunk,
                    policy_subchunk_text=policy_subchunk,
                    policy_chunk_text=policy_chunk,
                )

            addressed = bool(result.get("addressed"))
            policy_quote = result.get("policy_quote") if result.get("policy_quote") else None
            policy_text_for_binding = policy_subchunk or policy_chunk
            if addressed and policy_quote and not _citation_binding(policy_quote, policy_text_for_binding):
                addressed = False
                policy_quote = None
            missing = bool(result.get("missing")) or not addressed
            conflict = bool(result.get("conflict"))
            analysis_failed = not result

            if analysis_failed:
                status = "missing"
            elif conflict:
                status = "conflict"
            elif addressed:
                status = "addressed"
            else:
                status = "missing"

            requirement_summary = (
                truncate_at_sentence(statute_subchunk, 350) or "Requirement"
            ).strip()
            statute_quote = (
                truncate_at_sentence(statute_subchunk, self._config.max_statute_quote_chars).strip()
                or None
            )
            conflict_desc = result.get("conflict_description") if result else None
            if status == "missing" and not (conflict_desc or "").strip():
                conflict_desc = "The policy does not contain provisions that address this statutory requirement."
            gaps.append(
                GapItem(
                    jurisdiction=jurisdiction,
                    statute_reference=statute_ref,
                    statute_name=None,
                    statute_chunk_id=subchunk_id or None,
                    section=None,
                    requirement_summary=requirement_summary,
                    status=status,
                    policy_quote=policy_quote,
                    statute_quote=statute_quote,
                    conflict_description=conflict_desc,
                    analysis_failed=analysis_failed,
                    policy_subchunk_text=policy_subchunk or None,
                    policy_combined_sections=policy_chunk or None,
                    statute_subchunk_text=statute_subchunk or None,
                    statute_chunk_text=statute_chunk or None,
                )
            )

        total = len(gaps)
        summary = GapSummary(
            total_requirements=total,
            missing=sum(1 for g in gaps if g.status == "missing"),
            addressed=sum(1 for g in gaps if g.status == "addressed"),
            conflicts=sum(1 for g in gaps if g.status == "conflict"),
        )

        analyzed_at = _iso()
        retrieval_metadata = RetrievalMetadata(
            statute_subchunks_considered=statute_subchunks_considered,
            statute_pairs_matched=len(gaps),
        )
        response = GapAnalysisResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
            retrieval_metadata=retrieval_metadata,
            statute_chunk_ids_used=statute_chunk_ids_used[:500] if statute_chunk_ids_used else None,
            run_types=["gap"],
        )

        if getattr(req, "save_results", True):
            try:
                await self._storage.write_compliance_result({
                    "policy_document_id": policy_document_id,
                    "company_name": company_name,
                    "applicable_jurisdictions": jurisdictions,
                    "gaps": [g.model_dump() for g in gaps],
                    "summary": summary.model_dump(),
                    "analyzed_at": analyzed_at,
                    "run_types": ["gap"],
                })
                await self._storage.write_compliance_run_log({
                    "policy_document_id": policy_document_id,
                    "applicable_jurisdictions": jurisdictions,
                    "run_timestamp": analyzed_at,
                    "statute_chunk_ids_used": statute_chunk_ids_used[:500],
                })
            except Exception as e:
                import logging
                logging.getLogger("policy-compliance").warning("Failed to write compliance result/run_log: %s", e)

        return response

    async def gap_analysis_v2(self, req: GapAnalysisRequest) -> GapAnalysisResponse:
        """Chunk-level gap analysis: statute_embeddings vs policy_embeddings. Uses YAML-configurable prompt."""
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        if not policy_document_id:
            raise ComplianceSuiteServiceError("policy_document_id is required.", status_code=400)

        _, company_name = await self._load_policy_text(
            policy_document_id,
            database=req.database,
            policy_collection=req.policy_collection,
        )

        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        if not jurisdictions:
            appl = await self.applicability(
                ApplicabilityRequest(policy_document_id=policy_document_id, database=req.database, policy_collection=req.policy_collection)
            )
            jurisdictions = appl.applicable_jurisdictions or self._config.default_jurisdictions

        retrieve_result = await self._retriever.retrieve_policy_chunks_for_statute_chunks(
            database=self._retrieval_database,
            policy_document_id=policy_document_id,
            applicable_jurisdictions=jurisdictions,
            statute_document_id=req.statute_document_id,
            top_k_per_statute=1,
        )
        pairs = retrieve_result.pairs
        statute_chunks_considered = retrieve_result.statute_chunks_considered
        if req.num_rows is not None:
            pairs = pairs[: req.num_rows]

        if not pairs:
            doc_id_field = self._config.policy_document_id_field
            pol_coll = self._mongo_client[self._retrieval_database][
                self._config.policy_embeddings_collection
            ]

            def _count_policy_chunks():
                return pol_coll.count_documents({doc_id_field: policy_document_id})

            n = await _run_in_thread(_count_policy_chunks)
            if n == 0:
                raise ComplianceSuiteServiceError(
                    "Policy not indexed for gap analysis (v2). Index this policy into "
                    f"{self._config.policy_embeddings_collection} before running gap analysis.",
                    status_code=400,
                )

        emb_text = self._config.embedding_text_field
        pol_text_field = self._config.policy_chunk_text_field
        jur_field = self._config.statute_jurisdiction_field

        gaps: List[GapItem] = []
        seen: Set[Tuple[str, str]] = set()
        statute_chunk_ids_used: List[str] = []

        for pair in pairs:
            stat_doc = pair.statute_doc
            pol_doc = pair.policy_doc
            statute_chunk_text = (stat_doc.get(emb_text) or stat_doc.get("chunk_text") or "").strip()
            policy_chunk_text = (pol_doc.get(pol_text_field) or pol_doc.get("chunk_text") or "").strip()

            chunk_id = str(stat_doc.get("chunk_index") or stat_doc.get("_id") or "")
            statute_ref = str(stat_doc.get("document_id") or stat_doc.get("source_id") or "")
            jurisdiction = str(stat_doc.get(jur_field) or "")

            key = (statute_ref, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            statute_chunk_ids_used.append(f"{statute_ref}:{chunk_id}")

            async with self._semaphore:
                result = await self._llm_client.gap_check_chunks(
                    statute_chunk_text=statute_chunk_text,
                    policy_chunk_text=policy_chunk_text,
                )

            addressed = bool(result.get("addressed"))
            policy_quote = result.get("policy_quote") if result.get("policy_quote") else None
            if addressed and policy_quote and not _citation_binding(policy_quote, policy_chunk_text):
                addressed = False
                policy_quote = None
            missing = bool(result.get("missing")) or not addressed
            conflict = bool(result.get("conflict"))
            analysis_failed = not result

            if analysis_failed:
                status = "missing"
            elif conflict:
                status = "conflict"
            elif addressed:
                status = "addressed"
            else:
                status = "missing"

            requirement_summary = (
                truncate_at_sentence(statute_chunk_text, 350) or "Requirement"
            ).strip()
            statute_quote = (
                truncate_at_sentence(statute_chunk_text, self._config.max_statute_quote_chars).strip()
                or None
            )
            conflict_desc = result.get("conflict_description") if result else None
            if status == "missing" and not (conflict_desc or "").strip():
                conflict_desc = "The policy does not contain provisions that address this statutory requirement."
            gaps.append(
                GapItem(
                    jurisdiction=jurisdiction,
                    statute_reference=statute_ref,
                    statute_name=None,
                    statute_chunk_id=chunk_id or None,
                    section=None,
                    requirement_summary=requirement_summary,
                    status=status,
                    policy_quote=policy_quote,
                    statute_quote=statute_quote,
                    conflict_description=conflict_desc,
                    analysis_failed=analysis_failed,
                    policy_combined_sections=policy_chunk_text or None,
                    statute_chunk_text=statute_chunk_text or None,
                )
            )

        summary = GapSummary(
            total_requirements=len(gaps),
            missing=sum(1 for g in gaps if g.status == "missing"),
            addressed=sum(1 for g in gaps if g.status == "addressed"),
            conflicts=sum(1 for g in gaps if g.status == "conflict"),
        )

        analyzed_at = _iso()
        retrieval_metadata = RetrievalMetadata(
            statute_chunks_considered=statute_chunks_considered,
            statute_pairs_matched=len(gaps),
        )
        response = GapAnalysisResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
            retrieval_metadata=retrieval_metadata,
            statute_chunk_ids_used=statute_chunk_ids_used[:500] if statute_chunk_ids_used else None,
            run_types=["gap_v2"],
        )

        if getattr(req, "save_results", True):
            try:
                await self._storage.write_compliance_result({
                    "policy_document_id": policy_document_id,
                    "company_name": company_name,
                    "applicable_jurisdictions": jurisdictions,
                    "gaps": [g.model_dump() for g in gaps],
                    "summary": summary.model_dump(),
                    "analyzed_at": analyzed_at,
                    "run_types": ["gap_v2"],
                })
                await self._storage.write_compliance_run_log({
                    "policy_document_id": policy_document_id,
                    "applicable_jurisdictions": jurisdictions,
                    "run_timestamp": analyzed_at,
                    "statute_chunk_ids_used": statute_chunk_ids_used[:500],
                })
            except Exception as e:
                import logging
                logging.getLogger("policy-compliance").warning("Failed to write compliance result/run_log: %s", e)

        return response

    async def health_score(self, req: HealthScoreRequest) -> HealthScoreResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        gap_req = GapAnalysisRequest(
            policy_document_id=policy_document_id,
            applicable_jurisdictions=req.applicable_jurisdictions,
            database=req.database,
            policy_collection=req.policy_collection,
            save_results=req.save_results,
        )
        # Use v4 gap analysis (statute_sub_topic_embeddings, policy_legal_embeddings, consumer_rights/controller_duties)
        if self._gap_analysis_v4_service is not None:
            gap_result = await self._gap_analysis_v4_service.run(gap_req)
        else:
            gap_result = await self.gap_analysis(gap_req)

        if not gap_result.gaps:
            return HealthScoreResponse(
                policy_document_id=policy_document_id,
                company_name=gap_result.company_name,
                privacy_health_score=None,
                score_breakdown={},
                components={"requirements_total": 0, "addressed": 0, "missing": 0, "conflicts": 0},
                analyzed_at=gap_result.analyzed_at,
                error="no_applicable_statutes",
            )

        weights = req.weights or self._config.requirement_weights or {}
        total_weight = 0.0
        weighted_sum = 0.0
        j_weight_sum: Dict[str, float] = {}
        j_weight_addressed: Dict[str, float] = {}

        def _get_weight(requirement_summary: str) -> float:
            """Resolve weight: exact match on first 50 chars, else longest substring match, else 1.0."""
            exact_key = requirement_summary[:50]
            if exact_key in weights:
                return weights[exact_key]
            req_lower = requirement_summary.lower()
            # Sort keys by length descending so longer matches win (e.g. "right to delete" before "right to")
            for key in sorted(weights.keys(), key=len, reverse=True):
                if key.lower() in req_lower:
                    return weights[key]
            return 1.0

        for g in gap_result.gaps:
            if g.analysis_failed:
                continue
            w = _get_weight(g.requirement_summary)
            total_weight += w
            if g.status == "addressed":
                weighted_sum += w
            j_weight_sum[g.jurisdiction] = j_weight_sum.get(g.jurisdiction, 0.0) + w
            if g.status == "addressed":
                j_weight_addressed[g.jurisdiction] = j_weight_addressed.get(g.jurisdiction, 0.0) + w

        if total_weight <= 0:
            return HealthScoreResponse(
                policy_document_id=policy_document_id,
                company_name=gap_result.company_name,
                privacy_health_score=None,
                score_breakdown={},
                components={"requirements_total": len(gap_result.gaps), "addressed": gap_result.summary.addressed, "missing": gap_result.summary.missing, "conflicts": gap_result.summary.conflicts},
                analyzed_at=gap_result.analyzed_at,
                error="insufficient_analysis",
            )

        raw_ratio = weighted_sum / total_weight
        conflict_penalty = self._config.conflict_penalty_multiplier
        if gap_result.summary.conflicts and conflict_penalty < 1.0:
            raw_ratio *= conflict_penalty
        score = max(0, min(100, round(100 * raw_ratio)))

        by_jurisdiction_scores = {}
        for j, denom in j_weight_sum.items():
            num = j_weight_addressed.get(j, 0.0)
            if denom > 0:
                by_jurisdiction_scores[j] = max(0, min(100, round(100 * num / denom)))

        components = {
            "requirements_total": gap_result.summary.total_requirements,
            "addressed": gap_result.summary.addressed,
            "missing": gap_result.summary.missing,
            "conflicts": gap_result.summary.conflicts,
            "raw_ratio": round(raw_ratio, 4),
            "conflict_penalty_applied": bool(gap_result.summary.conflicts and conflict_penalty < 1.0),
        }

        response = HealthScoreResponse(
            policy_document_id=policy_document_id,
            company_name=gap_result.company_name,
            privacy_health_score=score,
            score_breakdown={"by_jurisdiction": by_jurisdiction_scores, "by_category": {}},
            components=components,
            analyzed_at=gap_result.analyzed_at,
        )

        if getattr(req, "save_results", True):
            try:
                await self._storage.write_compliance_result({
                    "policy_document_id": policy_document_id,
                    "company_name": gap_result.company_name,
                    "applicable_jurisdictions": gap_result.applicable_jurisdictions,
                    "privacy_health_score": score,
                    "score_breakdown": response.score_breakdown,
                    "components": components,
                    "analyzed_at": gap_result.analyzed_at,
                    "run_types": ["health_score"],
                })
            except Exception as e:
                import logging
                logging.getLogger("policy-compliance").warning("Failed to write health score result: %s", e)

        return response

    async def multi_jurisdictional(self, req: MultiJurisdictionalRequest) -> MultiJurisdictionalResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        jurisdictions = req.applicable_jurisdictions
        queries = self._config.disclosure_queries or ["right to know", "right to delete", "opt out of sale"]
        top_k_per_query = 3

        all_requirements_by_jurisdiction: Dict[str, List[Dict[str, str]]] = {}
        for j in jurisdictions:
            candidates = await self._retriever.retrieve_by_queries(
                database=self._retrieval_database,
                jurisdiction=j,
                queries=queries,
                top_k_per_query=top_k_per_query,
            )
            reqs: List[Dict[str, str]] = []
            for c in candidates[:10]:
                async with self._semaphore:
                    out = await self._llm_client.requirement_extraction(c.chunk_text)
                for r in out.get("requirements", [])[:3]:
                    reqs.append({"label": r.get("label", ""), "description": r.get("description", ""), "jurisdiction": j})
            all_requirements_by_jurisdiction[j] = reqs

        canonical_ids = self._config.canonical_requirement_ids or ["right_to_know", "right_to_delete", "opt_out_of_sale"]
        strictest_list: List[StrictestDenominatorItem] = []
        conflicts_list: List[ConflictBetweenJurisdictionsItem] = []

        for cid in canonical_ids[:5]:
            j_descriptions = []
            for j in jurisdictions:
                for r in all_requirements_by_jurisdiction.get(j, []):
                    if cid.replace("_", " ").lower() in (r.get("label", "") + r.get("description", "")).lower():
                        j_descriptions.append({"jurisdiction": j, "description": r.get("description", r.get("label", ""))})
                        break
            if not j_descriptions:
                continue
            async with self._semaphore:
                out = await self._llm_client.strictness_comparison(cid, j_descriptions)
            strictest_j = out.get("strictest_jurisdiction") or (j_descriptions[0]["jurisdiction"] if j_descriptions else "")
            strictest_desc = out.get("strictest_description") or ""
            strictest_list.append(
                StrictestDenominatorItem(
                    canonical_requirement_id=cid,
                    label=cid.replace("_", " ").title(),
                    strictest_jurisdiction=strictest_j,
                    strictest_description=strictest_desc,
                    all_jurisdictions=jurisdictions,
                    policy_alignment="not_provided",
                )
            )

        return MultiJurisdictionalResponse(
            applicable_jurisdictions=jurisdictions,
            analyzed_at=_iso(),
            strictest_common_denominator=strictest_list,
            conflicts_between_jurisdictions=conflicts_list,
        )

    async def drift_check(self, req: DriftCheckRequest) -> DriftCheckResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        last_at = await self._storage.get_last_drift_check_at()
        since_str = req.since
        if req.full_rebaseline:
            last_at = None
        elif since_str:
            try:
                last_at = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        db = self._mongo_client[self._database]
        statutes_coll = db[self._config.statutes_collection]
        version_field = getattr(self._config, "statute_index_version_field", "indexed_at")
        jurisdiction_field = self._config.statute_jurisdiction_field

        new_chunk_filter: Dict[str, Any] = {}
        if last_at is not None:
            last_val = last_at.isoformat() if hasattr(last_at, "isoformat") else str(last_at)
            new_chunk_filter[version_field] = {"$gt": last_val}

        def find_new_chunks():
            if not new_chunk_filter:
                return []
            cursor = statutes_coll.find(new_chunk_filter, {jurisdiction_field: 1, "_id": 1})
            return list(cursor)

        new_chunks = await _run_in_thread(find_new_chunks)
        affected_jurisdictions: List[str] = []
        if new_chunks:
            for d in new_chunks:
                j = d.get(jurisdiction_field)
                if j and j not in affected_jurisdictions:
                    affected_jurisdictions.append(j)

        policy_ids = req.policy_document_ids
        if not policy_ids:
            policy_ids = await self._storage.list_policy_document_ids(self._database)

        alerts: List[DriftAlertItem] = []
        policies_checked = 0

        for policy_document_id in policy_ids[:50]:
            policies_checked += 1
            last_result = await self._storage.get_last_compliance_result(policy_document_id)
            gap_req = GapAnalysisRequest(policy_document_id=policy_document_id, applicable_jurisdictions=affected_jurisdictions or self._config.default_jurisdictions)
            try:
                current_gap = await self.gap_analysis(gap_req)
            except Exception:
                continue
            if not last_result:
                continue
            prev_gaps = { (g.get("jurisdiction"), g.get("requirement_summary")): g for g in last_result.get("gaps", []) }
            curr_gaps = { (g.jurisdiction, g.requirement_summary): g for g in current_gap.gaps }
            new_gaps = [DriftGapItem(jurisdiction=g.jurisdiction, requirement_summary=g.requirement_summary, statute_reference=g.statute_reference) for k, g in curr_gaps.items() if k not in prev_gaps and g.status in ("missing", "conflict")]
            resolved_gaps = [DriftGapItem(jurisdiction=g.get("jurisdiction", ""), requirement_summary=g.get("requirement_summary", ""), statute_reference=g.get("statute_reference", "")) for k, g in prev_gaps.items() if k not in curr_gaps]

            current_score = None
            previous_score = last_result.get("privacy_health_score")
            try:
                hs = await self.health_score(HealthScoreRequest(policy_document_id=policy_document_id))
                current_score = hs.privacy_health_score
            except Exception:
                pass
            score_delta = (current_score - previous_score) if current_score is not None and previous_score is not None else None

            if new_gaps or (score_delta is not None and score_delta < 0):
                alert_id = str(uuid.uuid4())
                alerts.append(
                    DriftAlertItem(
                        alert_id=alert_id,
                        type="regulatory_drift",
                        policy_document_id=policy_document_id,
                        company_name=current_gap.company_name,
                        trigger="new_statute_indexed",
                        affected_jurisdictions=affected_jurisdictions or list(set(g.jurisdiction for g in current_gap.gaps)),
                        new_gaps=new_gaps,
                        resolved_gaps=resolved_gaps,
                        score_delta=score_delta,
                        previous_score=previous_score,
                        current_score=current_score,
                        detected_at=_iso(),
                    )
                )
                await self._storage.write_compliance_alert({
                    "alert_id": alert_id,
                    "type": "regulatory_drift",
                    "policy_document_id": policy_document_id,
                    "company_name": current_gap.company_name,
                    "trigger": "new_statute_indexed",
                    "affected_jurisdictions": affected_jurisdictions,
                    "new_gaps": [g.model_dump() for g in new_gaps],
                    "resolved_gaps": [g.model_dump() for g in resolved_gaps],
                    "score_delta": score_delta,
                    "previous_score": previous_score,
                    "current_score": current_score,
                    "detected_at": _iso(),
                })

        await self._storage.set_last_drift_check_at()

        return DriftCheckResponse(
            alerts=alerts,
            policies_checked=policies_checked,
            alerts_written=len(alerts),
        )

    def _build_report_markdown(
        self,
        result_data: Dict[str, Any],
        include_gap: bool = True,
        include_health_score: bool = True,
        include_multi_jurisdictional: bool = False,
    ) -> str:
        """Build markdown report from a compliance result dict (from storage or in-memory)."""
        lines = []
        policy_id = result_data.get("policy_document_id", "")
        company = result_data.get("company_name") or "—"
        analyzed_at = result_data.get("analyzed_at", "")
        lines.append(f"# Compliance Report")
        lines.append(f"**Policy ID:** {policy_id}")
        lines.append(f"**Company:** {company}")
        lines.append(f"**Analyzed at:** {analyzed_at}")
        lines.append("")

        if include_gap and result_data.get("gaps"):
            lines.append("## Gap Analysis")
            summary = result_data.get("summary") or {}
            lines.append(f"- Total requirements: {summary.get('total_requirements', 0)}")
            lines.append(f"- Addressed: {summary.get('addressed', 0)}")
            lines.append(f"- Missing: {summary.get('missing', 0)}")
            lines.append(f"- Conflicts: {summary.get('conflicts', 0)}")
            lines.append("")
            lines.append("| Jurisdiction | Statute | Section | Chunk ID | Requirement | Status | Policy quote | Conflict |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for g in result_data.get("gaps", []):
                req = (g.get("requirement_summary") or "")[:60]
                status = g.get("status", "")
                quote = (g.get("policy_quote") or "")[:40].replace("|", " ")
                conflict = (g.get("conflict_description") or "")[:40].replace("|", " ")
                section = (g.get("section") or "")[:30]
                chunk_id = g.get("statute_chunk_id") or ""
                statute_name = (g.get("statute_name") or "")[:20]
                lines.append(f"| {g.get('jurisdiction', '')} | {statute_name} | {section} | {chunk_id} | {req} | {status} | {quote} | {conflict} |")
            lines.append("")

        if include_health_score and result_data.get("privacy_health_score") is not None:
            lines.append("## Privacy Health Score")
            lines.append(f"**Score:** {result_data.get('privacy_health_score')}/100")
            breakdown = result_data.get("score_breakdown") or {}
            by_j = breakdown.get("by_jurisdiction") or {}
            if by_j:
                lines.append("**By jurisdiction:**")
                for j, s in by_j.items():
                    lines.append(f"- {j}: {s}")
            lines.append("")

        if include_multi_jurisdictional and result_data.get("strictest_common_denominator"):
            lines.append("## Strictest Common Denominator")
            for item in result_data.get("strictest_common_denominator", []):
                lines.append(f"- **{item.get('label', '')}** (strictest: {item.get('strictest_jurisdiction', '')})")
                lines.append(f"  {item.get('strictest_description', '')[:200]}")
            lines.append("")

        return "\n".join(lines)

    async def report(self, req: ReportRequest) -> Dict[str, Any]:
        """Generate report content (markdown or PDF). For PDF returns content_base64 or raises 501."""
        policy_document_id = (req.policy_document_id or "").strip()
        if not policy_document_id:
            raise ComplianceSuiteServiceError("policy_document_id is required.", status_code=400)

        if req.source == "latest_stored":
            doc = await self._storage.get_last_compliance_result(policy_document_id)
            if not doc:
                raise ComplianceSuiteServiceError("No stored result for this policy.", status_code=404)
            content = self._build_report_markdown(
                doc,
                include_gap=req.include_gap,
                include_health_score=req.include_health_score,
                include_multi_jurisdictional=req.include_multi_jurisdictional,
            )
        else:
            # run_now: run gap + health (and optionally multi-jurisdictional) without persisting
            gap_req = GapAnalysisRequest(
                policy_document_id=policy_document_id,
                applicable_jurisdictions=req.applicable_jurisdictions,
                save_results=False,
            )
            gap_result = await self.gap_analysis(gap_req)
            health_req = HealthScoreRequest(
                policy_document_id=policy_document_id,
                applicable_jurisdictions=req.applicable_jurisdictions,
                save_results=False,
            )
            health_result = await self.health_score(health_req)
            combined: Dict[str, Any] = {
                "policy_document_id": policy_document_id,
                "company_name": gap_result.company_name,
                "analyzed_at": gap_result.analyzed_at,
                "gaps": [g.model_dump() for g in gap_result.gaps],
                "summary": gap_result.summary.model_dump(),
                "privacy_health_score": health_result.privacy_health_score,
                "score_breakdown": health_result.score_breakdown,
                "components": health_result.components,
            }
            if req.include_multi_jurisdictional:
                multi_req = MultiJurisdictionalRequest(
                    applicable_jurisdictions=req.applicable_jurisdictions or gap_result.applicable_jurisdictions,
                )
                multi_result = await self.multi_jurisdictional(multi_req)
                combined["strictest_common_denominator"] = [
                    s.model_dump() for s in multi_result.strictest_common_denominator
                ]
            content = self._build_report_markdown(
                combined,
                include_gap=req.include_gap,
                include_health_score=req.include_health_score,
                include_multi_jurisdictional=req.include_multi_jurisdictional,
            )

        if req.format == "markdown":
            return {"format": "markdown", "content": content}
        # PDF: not implemented here; route may return 501 or convert via library
        raise ComplianceSuiteServiceError(
            "PDF generation not implemented. Use format=markdown.",
            status_code=501,
        )

    async def citations(self, req: CitationsRequest) -> CitationsResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        policy_text, company_name = await self._load_policy_text(policy_document_id)
        if not policy_text:
            raise ComplianceSuiteServiceError("Policy not found or empty.", status_code=404)

        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        queries = self._config.disclosure_queries or ["right to know", "right to delete", "opt out of sale"]
        top_k_per_query = max(2, (self._config.top_k_statutes or 5) // 2)

        citation_list: List[CitationItem] = []
        seen: Set[Tuple[str, str]] = set()

        for jurisdiction in jurisdictions:
            candidates = await self._retriever.retrieve_by_queries(
                database=self._retrieval_database,
                jurisdiction=jurisdiction,
                queries=queries,
                top_k_per_query=top_k_per_query,
            )
            for c in candidates[:15]:
                key = (c.statute_id or c.section_id or "", c.chunk_text[:80])
                if key in seen:
                    continue
                seen.add(key)
                async with self._semaphore:
                    out = await self._llm_client.citation_check(
                        policy_excerpt=policy_text[:3000],
                        statute_chunk_text=c.chunk_text,
                        statute_reference=c.statute_id or c.section_id or "",
                        jurisdiction=jurisdiction,
                    )
                citation_list.append(
                    CitationItem(
                        policy_excerpt=policy_text[:200],
                        policy_chunk_id=None,
                        statute_reference=c.statute_id or c.section_id or "",
                        jurisdiction=jurisdiction,
                        alignment=bool(out.get("alignment")),
                        statute_excerpt=out.get("statute_excerpt"),
                    )
                )

        aligned = sum(1 for x in citation_list if x.alignment)
        summary = CitationsSummary(
            total_citations=len(citation_list),
            aligned=aligned,
            not_aligned=len(citation_list) - aligned,
        )
        return CitationsResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=_iso(),
            citations=citation_list,
            summary=summary,
        )

    async def risk_assessment(self, req: RiskAssessmentRequest) -> RiskAssessmentResponse:
        if not await self._rate_limiter.allow():
            raise ComplianceSuiteServiceError("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        policy_text, company_name = await self._load_policy_text(policy_document_id)
        if not policy_text:
            raise ComplianceSuiteServiceError("Policy not found or empty.", status_code=404)

        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        queries = self._config.disclosure_queries or ["right to know", "right to delete", "opt out of sale"]
        top_k_per_query = 3
        statute_parts: List[str] = []
        for jurisdiction in jurisdictions:
            candidates = await self._retriever.retrieve_by_queries(
                database=self._retrieval_database,
                jurisdiction=jurisdiction,
                queries=queries,
                top_k_per_query=top_k_per_query,
            )
            for c in candidates[:5]:
                statute_parts.append(f"[{jurisdiction}] {c.chunk_text[:500]}")
        statute_summary = "\n\n".join(statute_parts)[:6000]

        async with self._semaphore:
            assessment = await self._llm_client.risk_assessment(policy_text, statute_summary)

        report_md = None
        if req.include_report:
            report_lines = [
                "# Risk Assessment",
                f"**Policy:** {policy_document_id}",
                "",
                "## Processing purposes",
                *["- " + p for p in (assessment.get("processing_purposes") or [])],
                "",
                "## Data categories",
                *["- " + cat for cat in (assessment.get("data_categories") or [])],
                "",
                "## Risks",
            ]
            for r in assessment.get("risks") or []:
                if isinstance(r, dict):
                    report_lines.append(f"- {r.get('description', '')} (severity: {r.get('severity', '')})")
                else:
                    report_lines.append(f"- {r}")
            report_lines.extend(["", "## Mitigations", *["- " + m for m in (assessment.get("mitigations") or [])]])
            report_md = "\n".join(report_lines)

        return RiskAssessmentResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            template_id=req.template_id or "default",
            analyzed_at=_iso(),
            assessment=assessment,
            report=report_md,
        )

    async def list_templates(self) -> TemplatesResponse:
        return TemplatesResponse(
            templates=[TemplateItem(id="default", label="Default DPIA-style")],
        )
