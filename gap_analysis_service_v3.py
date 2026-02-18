"""Gap analysis v3 service - implements GapAnalysisProcessDesign.md.

Flow: Statute items -> Vector search -> Policy matches -> LLM analysis per pair ->
Citation binding check -> Aggregation -> Optional persistence.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

from compliance_config import ComplianceConfig
from compliance_suite_schemas import (
    GapAnalysisRequest,
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    RetrievalMetadata,
)
from compliance_storage import ComplianceStorage
from compliance_utils import truncate_at_sentence, utc_now
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter
from vector_retriever import PolicyMatch, StatutePolicyPairV3, VectorRetriever


async def _run_in_thread(func, *args, **kwargs):
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _normalize(s: str) -> str:
    """Normalize whitespace and lowercase for citation binding comparison."""
    return " ".join((s or "").split()).lower()


def _citation_binding(policy_quote: Optional[str], policy_text: str) -> bool:
    """Return True if policy_quote is a substring of policy_text (normalized)."""
    if not policy_quote or not policy_text:
        return False
    return _normalize(policy_quote) in _normalize(policy_text)


class GapAnalysisServiceV3Error(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GapAnalysisServiceV3:
    """Gap analysis v3 per GapAnalysisProcessDesign.md."""

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
    ) -> tuple[str, Optional[str]]:
        """Load policy full text and company_name. Returns (text, company_name)."""
        db_name = database or self._database
        coll_name = policy_collection or self._config.policies_collection
        doc_id_field = self._config.policy_document_id_field

        def find():
            coll = self._mongo_client[db_name][coll_name]
            doc = coll.find_one({doc_id_field: policy_document_id})
            if not doc:
                from bson import ObjectId
                from bson.errors import InvalidId
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

    async def _policy_indexed(self, policy_document_id: str, database: str) -> bool:
        """Check if policy has at least one record in policy_embeddings."""
        coll = self._mongo_client[database][self._config.policy_embeddings_collection]
        doc_id_field = self._config.policy_document_id_field

        def count():
            return coll.count_documents({doc_id_field: policy_document_id})

        n = await _run_in_thread(count)
        return n > 0

    async def run(self, req: GapAnalysisRequest) -> GapAnalysisResponse:
        """Execute gap analysis v3 per design document."""
        if not await self._rate_limiter.allow():
            raise GapAnalysisServiceV3Error("Rate limit exceeded", status_code=429)

        policy_document_id = req.policy_document_id
        if not policy_document_id:
            raise GapAnalysisServiceV3Error("policy_document_id is required.", status_code=400)

        # Step 0: Validate policy exists and is indexed
        policy_text, company_name = await self._load_policy_text(
            policy_document_id,
            database=req.database,
            policy_collection=req.policy_collection,
        )

        indexed = await self._policy_indexed(policy_document_id, self._retrieval_database)
        if not indexed:
            raise GapAnalysisServiceV3Error(
                f"Policy not indexed for gap analysis. Index this policy into "
                f"{self._config.policy_embeddings_collection} before running gap analysis.",
                status_code=400,
            )

        # Resolve jurisdictions (design: use provided or applicability service or default)
        jurisdictions = req.applicable_jurisdictions or self._config.default_jurisdictions
        if not jurisdictions:
            jurisdictions = self._config.default_jurisdictions

        # Step 1 & 2: Retrieve statute items and policy matches via vector search
        top_k = getattr(self._config, "gap_analysis_v3_top_k", 5)
        num_candidates = getattr(self._config, "gap_analysis_v3_num_candidates", 50)
        score_threshold = getattr(self._config, "gap_analysis_v3_score_threshold", 0.70)

        pairs = await self._retriever.retrieve_statute_policy_pairs_v3(
            database=self._retrieval_database,
            policy_document_id=policy_document_id,
            applicable_jurisdictions=jurisdictions,
            statute_document_id=req.statute_document_id,
            top_k=top_k,
            num_candidates=num_candidates,
            score_threshold=score_threshold,
            num_rows=req.num_rows,
        )

        statute_items_considered = len(pairs)
        statute_pairs_matched = sum(1 for p in pairs if p.above_threshold_count > 0)

        # Load full policy text for citation binding (Step 4)
        if not policy_text:
            policy_text, _ = await self._load_policy_text(
                policy_document_id, database=req.database, policy_collection=req.policy_collection
            )

        jur_field = self._config.statute_jurisdiction_field
        emb_text = self._config.embedding_text_field
        pol_text_field = self._config.policy_chunk_text_field or "chunk_text"

        gaps: List[GapItem] = []
        seen: Set[str] = set()
        statute_item_ids_used: List[str] = []

        for pair in pairs:
            stat_doc = pair.statute_doc
            statute_text = (stat_doc.get(emb_text) or stat_doc.get("text") or "").strip()
            statute_ref = str(stat_doc.get("statute_reference") or stat_doc.get("document_id") or stat_doc.get("source_id") or "")
            jurisdiction = str(stat_doc.get(jur_field) or "")
            stat_id = str(stat_doc.get("_id", ""))

            key = f"{statute_ref}:{stat_id}"
            if key in seen:
                continue
            seen.add(key)
            statute_item_ids_used.append(stat_id)

            # Concatenate policy matches (above threshold first, then rest)
            policy_chunk_parts: List[str] = []
            for m in pair.policy_matches:
                if m.text:
                    policy_chunk_parts.append(m.text)
            policy_chunk = "\n\n---\n\n".join(policy_chunk_parts) if policy_chunk_parts else ""

            # Step 3: LLM analysis
            async with self._semaphore:
                result = await self._llm_client.gap_check_v3(
                    statute_chunk_text=statute_text,
                    policy_chunk_text=policy_chunk,
                )

            analysis_failed = result.get("_analysis_failed", False)
            status = result.get("status", "missing")
            policy_quote = result.get("policy_quote") if result.get("policy_quote") else None
            confidence = result.get("confidence", "low")
            citation_binding_failed: Optional[bool] = None

            # Step 4: Citation binding validation (against full policy text or concatenated matches)
            policy_text_for_binding = policy_text or policy_chunk
            if not analysis_failed and status in ("addressed", "conflict") and policy_quote and policy_text_for_binding:
                if not _citation_binding(policy_quote, policy_text_for_binding):
                    citation_binding_failed = True
                    if status == "addressed":
                        status = "missing"
                        policy_quote = None
                    # For conflict, keep status but flag the quote

            requirement_summary = (
                result.get("requirement_summary") or truncate_at_sentence(statute_text, 350) or "Requirement"
            ).strip()
            statute_quote = (
                truncate_at_sentence(
                    result.get("statute_quote") or statute_text,
                    getattr(self._config, "max_statute_quote_chars", 300),
                ).strip()
                or None
            )

            gaps.append(
                GapItem(
                    jurisdiction=jurisdiction,
                    statute_reference=statute_ref,
                    statute_name=None,
                    statute_chunk_id=stat_id or None,
                    section=None,
                    requirement_summary=requirement_summary,
                    status=status,
                    policy_quote=policy_quote,
                    statute_quote=statute_quote,
                    conflict_description=result.get("conflict_description") if result else None,
                    analysis_failed=analysis_failed,
                    confidence=confidence if confidence in ("high", "medium", "low") else None,
                    citation_binding_failed=citation_binding_failed,
                    statute_chunk_text=statute_text or None,
                    policy_chunk_text=policy_chunk or None,
                )
            )

        # Step 5: Aggregate
        summary = GapSummary(
            total_requirements=len(gaps),
            missing=sum(1 for g in gaps if g.status == "missing" and not g.analysis_failed),
            addressed=sum(1 for g in gaps if g.status == "addressed" and not g.analysis_failed),
            conflicts=sum(1 for g in gaps if g.status == "conflict" and not g.analysis_failed),
            analysis_failures=sum(1 for g in gaps if g.analysis_failed),
        )

        analyzed_at = utc_now().isoformat()
        retrieval_metadata = RetrievalMetadata(
            statute_items_considered=statute_items_considered,
            statute_pairs_matched=statute_pairs_matched,
        )

        response = GapAnalysisResponse(
            policy_document_id=policy_document_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
            retrieval_metadata=retrieval_metadata,
        )

        # Step 6: Persistence
        if getattr(req, "save_results", True):
            try:
                result_doc: Dict[str, Any] = {
                    "policy_document_id": policy_document_id,
                    "company_name": company_name,
                    "applicable_jurisdictions": jurisdictions,
                    "analyzed_at": analyzed_at,
                    "gaps": [g.model_dump() for g in gaps],
                    "summary": summary.model_dump(),
                    "retrieval_metadata": {
                        "statute_items_considered": statute_items_considered,
                        "statute_pairs_matched": statute_pairs_matched,
                    },
                    "run_type": "gap_analysis_v3",
                    "version": "v3",
                }
                await self._storage.write_compliance_result({
                    **result_doc,
                    "run_types": ["gap_v3"],
                })
                await self._storage.write_compliance_run_log({
                    "policy_document_id": policy_document_id,
                    "run_type": "gap_analysis_v3",
                    "statute_item_ids": statute_item_ids_used[:500],
                    "ran_at": analyzed_at,
                    "run_timestamp": analyzed_at,
                    "summary": summary.model_dump(),
                })
            except Exception as e:
                import logging
                logging.getLogger("policy-compliance").warning(
                    "Failed to write v3 compliance result/run_log: %s", e
                )

        return response
