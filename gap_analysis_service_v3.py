"""
Gap analysis v3 - SIMPLIFIED for easy debugging.
All logic in one place. No run_in_thread. Direct sync Mongo calls.
"""
from __future__ import annotations

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
from compliance_utils import jurisdiction_filter_values, normalize_jurisdiction, truncate_at_sentence, utc_now
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter

# For jurisdiction filter when none provided
VECTOR_SEARCH_JURISDICTION = "California"


def _normalize(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _citation_binding(policy_quote: Optional[str], policy_text: str) -> bool:
    if not policy_quote or not policy_text:
        return False
    return _normalize(policy_quote) in _normalize(policy_text)


class GapAnalysisServiceV3Error(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GapAnalysisServiceV3:
    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        retriever: Any,
        llm_client: AnthropicLLMClient,
        storage: ComplianceStorage,
        rate_limiter: RateLimiter,
    ) -> None:
        self.mongo = mongo_client
        self.cfg = config
        self.llm = llm_client
        self.storage = storage
        self.rate_limiter = rate_limiter
        self.db = (config.statute_database or "").strip() or config.compliance_database

    async def run(self, req: GapAnalysisRequest) -> GapAnalysisResponse:
        policy_doc_id = req.policy_document_id
        if not policy_doc_id:
            raise GapAnalysisServiceV3Error("policy_document_id is required.", status_code=400)

        if not await self.rate_limiter.allow():
            raise GapAnalysisServiceV3Error("Rate limit exceeded", status_code=429)

        # --- STEP 0: Load policy and check indexed ---
        db_for_policy = (req.database or "").strip() or self.cfg.compliance_database
        policies_coll = self.mongo[db_for_policy][self.cfg.policies_collection]
        policy_doc = policies_coll.find_one({self.cfg.policy_document_id_field: policy_doc_id})
        if not policy_doc:
            try:
                from bson import ObjectId
                policy_doc = policies_coll.find_one({"_id": ObjectId(policy_doc_id)})
            except Exception:
                policy_doc = policies_coll.find_one({"_id": policy_doc_id})
        if not policy_doc:
            raise GapAnalysisServiceV3Error("Policy not found", status_code=404)

        policy_text = (policy_doc.get("text") or "").strip()
        if not policy_text and policy_doc.get("policy_chunks"):
            parts = [c.get("chunk_text", "").strip() for c in policy_doc["policy_chunks"] if isinstance(c, dict)]
            policy_text = "\n\n".join(parts).strip()
        company_name = policy_doc.get("company_name") if isinstance(policy_doc.get("company_name"), str) else None

        policy_sub_coll = self.mongo[self.db][self.cfg.policy_sub_embeddings_collection]
        n_indexed = policy_sub_coll.count_documents({self.cfg.policy_document_id_field: policy_doc_id})
        if n_indexed == 0:
            raise GapAnalysisServiceV3Error(
                f"Policy not indexed. Index into {self.cfg.policy_sub_embeddings_collection} first.",
                status_code=400,
            )

        # --- STEP 1: Build jurisdiction filter and fetch statute subchunks ---
        jurisdictions = req.applicable_jurisdictions or self.cfg.default_jurisdictions or ["CA"]
        all_jur = []
        for j in jurisdictions:
            all_jur.extend(jurisdiction_filter_values(normalize_jurisdiction(j)))
        if not all_jur:
            all_jur = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)

        statute_filter = {self.cfg.statute_jurisdiction_field: {"$in": all_jur} if len(all_jur) > 1 else all_jur[0]}
        if req.statute_document_id:
            statute_filter["document_id"] = req.statute_document_id

        statute_sub_coll = self.mongo[self.db][self.cfg.statute_sub_embeddings_collection]
        statute_parent_coll = self.mongo[self.db][self.cfg.statute_embeddings_collection]
        policy_parent_coll = self.mongo[self.db][self.cfg.policy_embeddings_collection]

        vec_path = self.cfg.embedding_vector_field
        idx_name = self.cfg.vector_index_name
        stat_sub_f = self.cfg.statute_subchunk_text_field or "subchunk_text"
        pol_sub_f = self.cfg.policy_subchunk_text_field or "subchunk_text"
        pol_chunk_f = self.cfg.policy_chunk_text_field or "chunk_text"
        stat_chunk_f = self.cfg.statute_chunk_text_field or "chunk_text"
        emb_f = self.cfg.embedding_text_field
        doc_id_f = self.cfg.policy_document_id_field
        jur_f = self.cfg.statute_jurisdiction_field

        statute_docs = list(statute_sub_coll.find(
            statute_filter,
            {vec_path: 1, emb_f: 1, stat_sub_f: 1, "text": 1, "statute_reference": 1, jur_f: 1, "_id": 1, "parent_chunk_id": 1},
        ))
        if req.num_rows:
            statute_docs = statute_docs[: req.num_rows]

        top_k = getattr(self.cfg, "gap_analysis_v3_top_k", 5)
        score_thresh = getattr(self.cfg, "gap_analysis_v3_score_threshold", 0.70)
        policy_filter = {doc_id_f: policy_doc_id}

        gaps: List[GapItem] = []
        seen: Set[str] = set()
        statute_ids_used: List[str] = []

        # --- STEP 2 & 3: For each statute subchunk, vector search policy, fetch parents, LLM ---
        for stat_doc in statute_docs:
            qv = stat_doc.get(vec_path) or stat_doc.get("embedding")
            if not qv or not isinstance(qv, list):
                continue
            try:
                qv = [float(x) for x in qv]
            except (TypeError, ValueError):
                continue

            # Vector search policy subchunks
            limit = max(500, top_k * 500)
            vs = {"index": idx_name, "path": vec_path, "queryVector": qv, "numCandidates": limit, "limit": limit}
            pipe = [
                {"$vectorSearch": {**vs, "filter": policy_filter}},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$limit": top_k},
                {"$project": {"score": 1, pol_sub_f: 1, "text": 1, "parent_chunk_id": 1}},
            ]
            try:
                policy_results = list(policy_sub_coll.aggregate(pipe))
            except Exception:
                pipe_fb = [
                    {"$vectorSearch": vs},
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$match": policy_filter},
                    {"$limit": top_k},
                    {"$project": {"score": 1, pol_sub_f: 1, "text": 1, "parent_chunk_id": 1}},
                ]
                policy_results = list(policy_sub_coll.aggregate(pipe_fb))

            # Build policy matches with parent context
            policy_matches: List[tuple[str, str]] = []
            parent_ids = [r["parent_chunk_id"] for r in policy_results if r.get("parent_chunk_id") is not None]
            parent_map: Dict[str, str] = {}
            if parent_ids:
                for d in policy_parent_coll.find({"_id": {"$in": parent_ids}}, {"_id": 1, pol_chunk_f: 1, "text": 1}):
                    pid = d.get("_id")
                    if pid is not None:
                        parent_map[str(pid)] = (d.get(pol_chunk_f) or d.get("text") or "").strip()
            for r in policy_results:
                sub_txt = (r.get(pol_sub_f) or r.get("text") or "").strip()
                parent_txt = parent_map.get(str(r.get("parent_chunk_id") or ""), "")
                policy_matches.append((sub_txt, parent_txt))

            # Statute parent context
            stat_parent_id = stat_doc.get("parent_chunk_id")
            statute_parent_txt = ""
            if stat_parent_id:
                pdoc = statute_parent_coll.find_one({"_id": stat_parent_id}, {stat_chunk_f: 1, emb_f: 1, "text": 1})
                if pdoc:
                    statute_parent_txt = (pdoc.get(stat_chunk_f) or pdoc.get(emb_f) or pdoc.get("text") or "").strip()

            statute_subchunk = (stat_doc.get(stat_sub_f) or stat_doc.get(emb_f) or stat_doc.get("text") or "").strip()
            statute_ref = str(stat_doc.get("statute_reference") or stat_doc.get("document_id") or stat_doc.get("source_id") or "")
            jurisdiction = str(stat_doc.get(jur_f) or "")
            stat_id = str(stat_doc.get("_id", ""))

            key = f"{statute_ref}:{stat_id}"
            if key in seen:
                continue
            seen.add(key)
            statute_ids_used.append(stat_id)

            # Build blocks for LLM (parent + subchunk)
            # statute_parts = [p for p in [statute_parent_txt, statute_subchunk] if p]
            statute_parts = [p for p in [statute_subchunk] if p]
            statute_block = "\n\n".join(statute_parts) if statute_parts else statute_subchunk or ""
            policy_parts = []
            for sub_txt, parent_txt in policy_matches:
                if sub_txt or parent_txt:
                    policy_parts.append("\n\n".join([p for p in [parent_txt, sub_txt] if p]) if (parent_txt or sub_txt) else sub_txt or "")
            policy_block = "\n\n---\n\n".join(policy_parts) if policy_parts else ""

            # LLM call
            result = await self.llm.gap_check_v3(statute_chunk_text=statute_block, policy_chunk_text=policy_block)

            analysis_failed = result.get("_analysis_failed", False)
            status = result.get("status", "missing")
            policy_quote = result.get("policy_quote") if result.get("policy_quote") else None
            confidence = result.get("confidence", "low")
            citation_binding_failed: Optional[bool] = None

            # Citation binding
            binding_text = policy_text or policy_block
            if not analysis_failed and status in ("addressed", "conflict") and policy_quote and binding_text:
                if not _citation_binding(policy_quote, binding_text):
                    citation_binding_failed = True
                    if status == "addressed":
                        status = "missing"
                        policy_quote = None

            req_summary = (result.get("requirement_summary") or truncate_at_sentence(statute_block, 350) or "Requirement").strip()
            stat_quote = (truncate_at_sentence(result.get("statute_quote") or statute_block, getattr(self.cfg, "max_statute_quote_chars", 300)).strip() or None)

            policy_subchunk_joined = "\n\n---\n\n".join([m[0] for m in policy_matches if m[0]]) if policy_matches else None

            conflict_desc = result.get("conflict_description") if result else None
            if status == "missing" and not (conflict_desc or "").strip():
                conflict_desc = "The policy does not contain provisions that address this statutory requirement."

            gaps.append(
                GapItem(
                    jurisdiction=jurisdiction,
                    statute_reference=statute_ref,
                    statute_name=None,
                    statute_chunk_id=stat_id or None,
                    section=None,
                    requirement_summary=req_summary,
                    status=status,
                    policy_quote=policy_quote,
                    statute_quote=stat_quote,
                    conflict_description=conflict_desc,
                    analysis_failed=analysis_failed,
                    confidence=confidence if confidence in ("high", "medium", "low") else None,
                    citation_binding_failed=citation_binding_failed,
                    statute_subchunk_text=statute_subchunk or None,
                    statute_chunk_text=statute_block or None,
                    policy_subchunk_text=policy_subchunk_joined,
                    policy_combined_sections=policy_block or None,
                )
            )

        # --- STEP 4: Aggregate ---
        summary = GapSummary(
            total_requirements=len(gaps),
            missing=sum(1 for g in gaps if g.status == "missing" and not g.analysis_failed),
            addressed=sum(1 for g in gaps if g.status == "addressed" and not g.analysis_failed),
            conflicts=sum(1 for g in gaps if g.status == "conflict" and not g.analysis_failed),
            analysis_failures=sum(1 for g in gaps if g.analysis_failed),
        )
        analyzed_at = utc_now().isoformat()
        retrieval_metadata = RetrievalMetadata(
            statute_items_considered=len(statute_docs),
            statute_pairs_matched=len(gaps),
        )

        response = GapAnalysisResponse(
            policy_document_id=policy_doc_id,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
            retrieval_metadata=retrieval_metadata,
        )

        # --- STEP 5: Persist ---
        if getattr(req, "save_results", True):
            try:
                import uuid
                res_coll = self.mongo[db_for_policy][self.cfg.compliance_results_collection]
                log_coll = self.mongo[db_for_policy][self.cfg.compliance_run_log_collection]
                res_coll.insert_one({
                    "_id": str(uuid.uuid4()),
                    "policy_document_id": policy_doc_id,
                    "company_name": company_name,
                    "applicable_jurisdictions": jurisdictions,
                    "analyzed_at": analyzed_at,
                    "gaps": [g.model_dump() for g in gaps],
                    "summary": summary.model_dump(),
                    "retrieval_metadata": {"statute_items_considered": len(statute_docs), "statute_pairs_matched": len(gaps)},
                    "run_type": "gap_analysis_v3",
                    "version": "v3",
                    "run_types": ["gap_v3"],
                })
                log_coll.insert_one({
                    "policy_document_id": policy_doc_id,
                    "run_type": "gap_analysis_v3",
                    "statute_item_ids": statute_ids_used[:500],
                    "ran_at": analyzed_at,
                    "run_timestamp": analyzed_at,
                    "summary": summary.model_dump(),
                })
            except Exception as e:
                import logging
                logging.getLogger("policy-compliance").warning("Failed to write v3 result: %s", e)

        return response
