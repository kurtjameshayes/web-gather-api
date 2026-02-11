"""Service layer for the compliance suite: applicability, gap analysis, health score, multi-jurisdictional, drift."""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from compliance_config import ComplianceConfig
from compliance_suite_schemas import (
    ApplicabilityResponse,
    ConflictBetweenJurisdictionsItem,
    DriftAlertItem,
    DriftCheckResponse,
    DriftGapItem,
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    HealthScoreResponse,
    MultiJurisdictionalResponse,
    StrictestDenominatorItem,
)
from compliance_storage import ComplianceStorage
from compliance_utils import utc_now
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter
from vector_retriever import StatuteCandidate, VectorRetriever


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


class ComplianceSuiteService:
    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        retriever: VectorRetriever,
        llm_client: AnthropicLLMClient,
        storage: ComplianceStorage,
        rate_limiter: RateLimiter,
    ) -> None:
        self._mongo_client = mongo_client
        self._config = config
        self._retriever = retriever
        self._llm_client = llm_client
        self._storage = storage
        self._rate_limiter = rate_limiter
        self._semaphore = asyncio.Semaphore(config.llm_concurrency)
        self._database = config.compliance_database
        self._retrieval_database = (config.statute_database or "").strip() or config.compliance_database

    async def _load_policy_text(
        self,
        policy_document_id: str,
        database: Optional[str] = None,
        policy_collection: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Load policy full text and optional company_name. Returns (text, company_name)."""
        db_name = database or self._database
        coll_name = policy_collection or self._config.policies_collection

        def find():
            coll = self._mongo_client[db_name][coll_name]
            return coll.find_one({"_id": policy_document_id})

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

        policy_text, company_name = await self._load_policy_text(
            policy_document_id,
            database=req.database,
            policy_collection=req.policy_collection,
        )
        if not policy_text:
            raise ComplianceSuiteServiceError("Policy not found or empty.", status_code=404)

        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        if not jurisdictions:
            appl = await self.applicability(
                ApplicabilityRequest(policy_document_id=policy_document_id, database=req.database, policy_collection=req.policy_collection)
            )
            jurisdictions = appl.applicable_jurisdictions or self._config.default_jurisdictions

        queries = self._config.disclosure_queries or [
            "right to know what personal information is collected",
            "right to delete personal information",
            "opt out of sale of personal data",
        ]
        top_k_per_query = max(2, (self._config.top_k_statutes or 5) // 2)

        gaps: List[GapItem] = []
        seen: Set[Tuple[str, str]] = set()
        statute_chunk_ids_used: List[str] = []

        for jurisdiction in jurisdictions:
            candidates = await self._retriever.retrieve_by_queries(
                database=self._retrieval_database,
                jurisdiction=jurisdiction,
                queries=queries,
                top_k_per_query=top_k_per_query,
            )
            for c in candidates:
                key = (c.statute_id, c.chunk_header_text or c.chunk_text[:80])
                if key in seen:
                    continue
                seen.add(key)
                statute_chunk_ids_used.append(f"{c.statute_id}:{c.chunk_id}")

                async with self._semaphore:
                    result = await self._llm_client.gap_check(
                        statute_chunk_text=c.chunk_text,
                        statute_reference=c.statute_id or c.section_id,
                        policy_text=policy_text,
                    )

                addressed = bool(result.get("addressed"))
                policy_quote = result.get("policy_quote")
                if addressed and policy_quote and not _citation_binding(policy_quote, policy_text):
                    addressed = False
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

                requirement_summary = (c.chunk_header_text or c.chunk_text[:120] or "Requirement").strip()
                gaps.append(
                    GapItem(
                        jurisdiction=jurisdiction,
                        statute_reference=c.statute_id or c.section_id or "",
                        requirement_summary=requirement_summary,
                        status=status,
                        policy_quote=policy_quote if addressed else None,
                        conflict_description=result.get("conflict_description") if result else None,
                        analysis_failed=analysis_failed,
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
        response = GapAnalysisResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
        )

        try:
            await self._storage.write_compliance_result({
                "policy_document_id": policy_document_id,
                "company_name": company_name,
                "applicable_jurisdictions": jurisdictions,
                "gaps": [g.model_dump() for g in gaps],
                "summary": summary.model_dump(),
                "analyzed_at": analyzed_at,
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
        )
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
        for g in gap_result.gaps:
            if g.analysis_failed:
                continue
            w = weights.get(g.requirement_summary[:50], 1.0)
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

        try:
            await self._storage.write_compliance_result({
                "policy_document_id": policy_document_id,
                "company_name": gap_result.company_name,
                "applicable_jurisdictions": gap_result.applicable_jurisdictions,
                "privacy_health_score": score,
                "score_breakdown": response.score_breakdown,
                "components": components,
                "analyzed_at": gap_result.analyzed_at,
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
