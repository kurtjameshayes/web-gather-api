"""
Gap analysis v4 - Category-mapping-driven approach.
Uses statute_sub_topic_embeddings, policy_legal_embeddings, and category_mapping.
Single Python file, easy-to-follow POC.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from compliance_config import ComplianceConfig
from compliance_suite_schemas import (
    GapAnalysisRequest,
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    RetrievalMetadata,
)
from compliance_utils import truncate_at_sentence, utc_now
from llm_client import AnthropicLLMClient
from rate_limiter import RateLimiter

if TYPE_CHECKING:
    from adaptive_feedback_service import CriticService

logger = logging.getLogger("policy-compliance")

# Statute categories we analyze for compliance (not just context)
ANALYZE_CATEGORIES = ("consumer_rights", "controller_duties")
# Statute categories used only as reference context
CONTEXT_CATEGORIES = ("definitions", "applicability")


def _normalize(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _citation_binding(policy_quote: Optional[str], policy_text: str) -> bool:
    if not policy_quote or not policy_text:
        return False
    return _normalize(policy_quote) in _normalize(policy_text)


class GapAnalysisServiceV4Error(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GapAnalysisServiceV4:
    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        llm_client: AnthropicLLMClient,
        rate_limiter: RateLimiter,
        critic: Optional["CriticService"] = None,
    ) -> None:
        self.mongo = mongo_client
        self.cfg = config
        self.llm = llm_client
        self.rate_limiter = rate_limiter
        self.critic = critic
        self._cfg_enabled = config.adaptive_feedback_enabled
        self.db = (config.statute_database or "").strip() or config.compliance_database

    def _load_policy_doc(self, policies_coll: Any, policy_id: str) -> Optional[Dict[str, Any]]:
        doc_id_f = self.cfg.policy_document_id_field
        doc = policies_coll.find_one({doc_id_f: policy_id})
        if not doc:
            try:
                from bson import ObjectId
                doc = policies_coll.find_one({"_id": ObjectId(policy_id)})
            except Exception:
                doc = policies_coll.find_one({"_id": policy_id})
        return doc

    async def run(self, req: GapAnalysisRequest) -> GapAnalysisResponse:
        policy_ids = req.policy_document_ids if req.policy_document_ids else ([req.policy_document_id] if req.policy_document_id else [])
        if not policy_ids:
            raise GapAnalysisServiceV4Error("Either policy_document_id or policy_document_ids (non-empty) is required.", status_code=400)

        if not await self.rate_limiter.allow():
            raise GapAnalysisServiceV4Error("Rate limit exceeded", status_code=429)

        # --- STEP 0: Load policy docs and check indexed in policy_legal_embeddings ---
        db_for_policy = (req.database or "").strip() or self.cfg.compliance_database
        policies_coll = self.mongo[db_for_policy][self.cfg.policies_collection]
        policy_text_parts: List[str] = []
        company_name: Optional[str] = None
        for pid in policy_ids:
            policy_doc = self._load_policy_doc(policies_coll, pid)
            if not policy_doc:
                raise GapAnalysisServiceV4Error(f"Policy not found: {pid}", status_code=404)
            txt = (policy_doc.get("text") or "").strip()
            if not txt and policy_doc.get("policy_chunks"):
                parts = [c.get("chunk_text", "").strip() for c in policy_doc["policy_chunks"] if isinstance(c, dict)]
                txt = "\n\n".join(parts).strip()
            if txt:
                policy_text_parts.append(txt)
            if company_name is None and isinstance(policy_doc.get("company_name"), str):
                company_name = policy_doc.get("company_name")
        policy_text = "\n\n---\n\n".join(policy_text_parts).strip()

        doc_id_f = self.cfg.policy_document_id_field
        statute_coll = self.mongo[self.db][self.cfg.statute_sub_topic_embeddings_collection]
        policy_legal_coll = self.mongo[self.db][self.cfg.policy_legal_embeddings_collection]
        cat_map_coll = self.mongo[self.db][self.cfg.category_mapping_collection]

        n_indexed = policy_legal_coll.count_documents({doc_id_f: {"$in": policy_ids}})
        if n_indexed == 0:
            raise GapAnalysisServiceV4Error(
                f"Policy not indexed. Index into {self.cfg.policy_legal_embeddings_collection} first.",
                status_code=400,
            )

        # --- STEP 1: Load category mappings (consumer_rights, controller_duties only) ---
        mappings = list(cat_map_coll.find({"statute_category": {"$in": list(ANALYZE_CATEGORIES)}}))
        # Fallback: user spec said "category_mappings" (plural); db.py uses "category_mapping" (singular)
        if not mappings and self.cfg.category_mapping_collection == "category_mapping":
            alt_coll = self.mongo[self.db]["category_mappings"]
            mappings = list(alt_coll.find({"statute_category": {"$in": list(ANALYZE_CATEGORIES)}}))
            if mappings:
                cat_map_coll = alt_coll
        if not mappings:
            return self._empty_response(policy_ids, company_name, req, db_for_policy)

        jurisdictions = req.applicable_jurisdictions or self.cfg.default_jurisdictions or ["CA"]
        policy_filter = {doc_id_f: {"$in": policy_ids}} if len(policy_ids) > 1 else {doc_id_f: policy_ids[0]}

        # --- STEP 1b: Retrieve adaptive feedback from prior runs ---
        adaptive_feedback_text = ""
        feedback_ids_used: List[str] = []
        if self.critic and self.cfg.adaptive_feedback_enabled:
            try:
                feedback_docs = self.critic.get_active_feedback(policy_ids[0])
                if feedback_docs:
                    adaptive_feedback_text = self.critic.format_feedback_for_prompt(feedback_docs)
                    feedback_ids_used = [d["_id"] for d in feedback_docs]
                    logger.info(
                        "Injecting %d adaptive feedback items into v4 gap analysis for policy %s",
                        len(feedback_ids_used), policy_ids[0],
                    )
            except Exception as exc:
                logger.warning("Failed to retrieve adaptive feedback: %s", exc)

        gaps: List[GapItem] = []
        seen: Set[str] = set()
        statute_ids_used: List[str] = []
        statute_items_considered = 0

        # --- STEP 2 & 3: For each mapping, fetch statute docs, context, policy chunks, LLM ---
        for mapping in mappings:
            statute_category = mapping.get("statute_category")
            policy_categories = mapping.get("policy_categories") or []
            sub_topic = mapping.get("sub_topic")

            if not policy_categories or statute_category not in ANALYZE_CATEGORIES:
                continue

            # Fetch statute requirements matching this mapping
            stat_filter: Dict[str, Any] = {"category": statute_category}
            if sub_topic is not None and str(sub_topic).strip():
                stat_filter["sub_topic"] = str(sub_topic).strip()

            statute_docs = list(statute_coll.find(stat_filter))
            if req.num_rows and statute_items_considered + len(statute_docs) > req.num_rows:
                statute_docs = statute_docs[: max(0, req.num_rows - statute_items_considered)]

            for stat_doc in statute_docs:
                statute_items_considered += 1
                stat_id = str(stat_doc.get("_id", ""))
                document_id = stat_doc.get("document_id") or stat_doc.get("statute_reference") or stat_doc.get("source_id")
                header_text = (stat_doc.get("header_text") or "").strip()
                subtopic_text = (stat_doc.get("subtopic_text") or "").strip()
                requirement_summary = (stat_doc.get("requirement_summary") or "").strip()
                jurisdiction = str(stat_doc.get("jurisdiction") or jurisdictions[0] if jurisdictions else "")

                # Build statutory requirement block
                statutory_parts = [p for p in [header_text, subtopic_text or requirement_summary] if p]
                statutory_requirement = "\n\n".join(statutory_parts) if statutory_parts else (requirement_summary or subtopic_text or "Requirement")

                key = f"{statute_category}:{sub_topic or ''}:{stat_id}"
                if key in seen:
                    continue
                seen.add(key)
                statute_ids_used.append(stat_id)

                # Fetch definitions + applicability from same statute (document_id)
                reference_parts: List[str] = []
                if document_id is not None:
                    context_docs = list(statute_coll.find(
                        {"document_id": document_id, "category": {"$in": list(CONTEXT_CATEGORIES)}},
                        {"subtopic_text": 1, "requirement_summary": 1, "header_text": 1, "category": 1},
                    ))
                    for ctx in context_docs:
                        cat = ctx.get("category", "")
                        txt = (ctx.get("subtopic_text") or ctx.get("requirement_summary") or "").strip()
                        hdr = (ctx.get("header_text") or "").strip()
                        if txt or hdr:
                            sep = "\n\n"
                            block = f"[{cat}]\n{(hdr + sep + txt).strip()}" if hdr else txt
                            reference_parts.append(block)
                reference_context = "\n\n---\n\n".join(reference_parts) if reference_parts else "(No definitions or applicability context available.)"

                # Fetch policy chunks for mapped categories
                pol_filter = {**policy_filter, "category": {"$in": policy_categories}}
                policy_docs = list(policy_legal_coll.find(pol_filter, {"chunk_text": 1}))
                policy_chunks = [(d.get("chunk_text") or "").strip() for d in policy_docs if (d.get("chunk_text") or "").strip()]
                policy_block = "\n\n---\n\n".join(policy_chunks) if policy_chunks else ""

                # LLM call
                result = await self.llm.gap_check_v4(
                    reference_context=reference_context,
                    statutory_requirement=statutory_requirement,
                    policy_text=policy_block,
                    adaptive_feedback=adaptive_feedback_text,
                )

                analysis_failed = result.get("_analysis_failed", False)
                status = result.get("status", "missing")
                policy_quote = result.get("policy_quote") if result.get("policy_quote") else None
                confidence = result.get("confidence", "low")
                citation_binding_failed: Optional[bool] = None

                # Citation binding
                binding_text = policy_text or policy_block
                if not analysis_failed and status in ("addressed", "conflict", "partial", "ambiguous") and policy_quote and binding_text:
                    if not _citation_binding(policy_quote, binding_text):
                        citation_binding_failed = True
                        if status == "addressed":
                            status = "missing"
                            policy_quote = None

                req_summary = (result.get("requirement_summary") or requirement_summary or truncate_at_sentence(statutory_requirement, 350) or "Requirement").strip()
                stat_quote = (truncate_at_sentence(result.get("statute_quote") or statutory_requirement, getattr(self.cfg, "max_statute_quote_chars", 300)).strip() or None)

                conflict_desc = result.get("conflict_description")
                if status == "missing" and not (conflict_desc or "").strip():
                    conflict_desc = "The policy does not contain provisions that address this statutory requirement."

                statute_ref = header_text or str(document_id) or stat_id

                gaps.append(
                    GapItem(
                        jurisdiction=jurisdiction,
                        statute_reference=statute_ref,
                        statute_name=None,
                        statute_chunk_id=stat_id or None,
                        section=header_text or None,
                        requirement_summary=req_summary,
                        status=status,
                        policy_quote=policy_quote,
                        statute_quote=stat_quote,
                        conflict_description=conflict_desc,
                        analysis_failed=analysis_failed,
                        confidence=confidence if confidence in ("high", "medium", "low") else None,
                        citation_binding_failed=citation_binding_failed,
                        statute_subchunk_text=subtopic_text or None,
                        statute_chunk_text=statutory_requirement or None,
                        policy_subchunk_text=policy_block or None,
                        policy_combined_sections=policy_block or None,
                    )
                )

        # --- STEP 4: Aggregate ---
        summary = GapSummary(
            total_requirements=len(gaps),
            missing=sum(1 for g in gaps if g.status == "missing" and not g.analysis_failed),
            addressed=sum(1 for g in gaps if g.status == "addressed" and not g.analysis_failed),
            conflicts=sum(1 for g in gaps if g.status == "conflict" and not g.analysis_failed),
            partial=sum(1 for g in gaps if g.status == "partial" and not g.analysis_failed),
            ambiguous=sum(1 for g in gaps if g.status == "ambiguous" and not g.analysis_failed),
            analysis_failures=sum(1 for g in gaps if g.analysis_failed),
        )
        analyzed_at = utc_now().isoformat()
        retrieval_metadata = RetrievalMetadata(
            statute_items_considered=statute_items_considered,
            statute_pairs_matched=len(gaps),
        )

        policy_doc_id_display = policy_ids[0] if policy_ids else ""
        used_list = req.policy_document_ids is not None and len(req.policy_document_ids) > 0
        response = GapAnalysisResponse(
            policy_document_id=policy_doc_id_display,
            policy_document_ids=policy_ids if used_list and len(policy_ids) > 1 else None,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=analyzed_at,
            gaps=gaps,
            summary=summary,
            retrieval_metadata=retrieval_metadata,
            statute_chunk_ids_used=statute_ids_used[:500] if statute_ids_used else None,
            run_types=["gap_v4"],
            run_type="gap_analysis_v4",
            version="v4",
        )

        # --- STEP 5: Persist ---
        run_id: Optional[str] = None
        if getattr(req, "save_results", True):
            try:
                import uuid as _uuid
                run_id = str(_uuid.uuid4())
                res_coll = self.mongo[db_for_policy][self.cfg.compliance_results_collection]
                log_coll = self.mongo[db_for_policy][self.cfg.compliance_run_log_collection]
                res_doc: Dict[str, Any] = {
                    "_id": run_id,
                    "policy_document_id": policy_doc_id_display,
                    "company_name": company_name,
                    "applicable_jurisdictions": jurisdictions,
                    "analyzed_at": analyzed_at,
                    "gaps": [g.model_dump() for g in gaps],
                    "summary": summary.model_dump(),
                    "retrieval_metadata": {"statute_items_considered": statute_items_considered, "statute_pairs_matched": len(gaps)},
                    "run_type": "gap_analysis_v4",
                    "version": "v4",
                    "run_types": ["gap_v4"],
                }
                if used_list and len(policy_ids) > 1:
                    res_doc["policy_document_ids"] = policy_ids
                res_coll.insert_one(res_doc)
                log_doc: Dict[str, Any] = {
                    "policy_document_id": policy_doc_id_display,
                    "run_type": "gap_analysis_v4",
                    "statute_item_ids": statute_ids_used[:500],
                    "ran_at": analyzed_at,
                    "run_timestamp": analyzed_at,
                    "summary": summary.model_dump(),
                }
                if used_list and len(policy_ids) > 1:
                    log_doc["policy_document_ids"] = policy_ids
                log_coll.insert_one(log_doc)
            except Exception as e:
                logger.warning("Failed to write v4 result: %s", e)

        # --- STEP 6: Record feedback usage and trigger critic evaluation ---
        if self.critic and self._cfg_enabled and run_id:
            try:
                if feedback_ids_used:
                    await self.critic.record_feedback_usage(
                        run_id=run_id,
                        feedback_ids=feedback_ids_used,
                        rendered_text=adaptive_feedback_text,
                    )
                await self.critic.evaluate(response, run_id)
            except Exception as exc:
                logger.warning("Adaptive feedback step failed for run %s: %s", run_id, exc)

        return response

    def _empty_response(
        self,
        policy_ids: List[str],
        company_name: Optional[str],
        req: GapAnalysisRequest,
        db_for_policy: str,
    ) -> GapAnalysisResponse:
        """Return empty response when no category mappings exist."""
        jurisdictions = req.applicable_jurisdictions or self.cfg.default_jurisdictions or ["CA"]
        policy_doc_id_display = policy_ids[0] if policy_ids else ""
        used_list = req.policy_document_ids is not None and len(req.policy_document_ids) > 0
        return GapAnalysisResponse(
            policy_document_id=policy_doc_id_display,
            policy_document_ids=policy_ids if used_list and len(policy_ids) > 1 else None,
            company_name=company_name,
            applicable_jurisdictions=jurisdictions,
            analyzed_at=utc_now().isoformat(),
            gaps=[],
            summary=GapSummary(),
            retrieval_metadata=RetrievalMetadata(statute_items_considered=0, statute_pairs_matched=0),
            statute_chunk_ids_used=None,
            run_types=["gap_v4"],
            run_type="gap_analysis_v4",
            version="v4",
        )
