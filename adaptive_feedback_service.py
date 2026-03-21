"""Adaptive feedback service: critic evaluation and feedback injection for V4 gap analysis."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from async_utils import run_in_thread
from compliance_config import ComplianceConfig
from compliance_suite_schemas import GapAnalysisResponse
from compliance_utils import utc_now

logger = logging.getLogger("policy-compliance")

VALID_CATEGORIES = frozenset({
    "false_positive",
    "false_negative",
    "citation_quality",
    "confidence_calibration",
    "prompt_guidance",
})
VALID_SEVERITIES = frozenset({"high", "medium", "low"})


class CriticService:
    """Evaluates gap analysis runs and manages the adaptive feedback loop.

    All writes to ``adaptive_feedback`` are append-only (immutable documents).
    Older feedback for the same policy is linked via ``superseded_by`` but
    never deleted, preserving a complete audit trail.
    """

    def __init__(
        self,
        mongo_client: Any,
        config: ComplianceConfig,
        llm_client: Any,
    ) -> None:
        self._mongo = mongo_client
        self._cfg = config
        self._llm = llm_client
        self._db_name = config.compliance_database

    # ------------------------------------------------------------------
    # MongoDB helpers
    # ------------------------------------------------------------------

    def _feedback_coll(self):
        return self._mongo[self._db_name][self._cfg.adaptive_feedback_collection]

    def _log_coll(self):
        return self._mongo[self._db_name][self._cfg.adaptive_feedback_log_collection]

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create indexes for efficient querying (idempotent)."""
        try:
            self._feedback_coll().create_index(
                [("policy_document_id", 1), ("superseded_by", 1), ("created_at", -1)],
                name="feedback_policy_active_idx",
            )
            self._log_coll().create_index(
                [("run_id", 1)],
                name="feedback_log_run_idx",
            )
            self._log_coll().create_index(
                [("feedback_ids_used", 1)],
                name="feedback_log_ids_used_idx",
            )
            logger.info("Ensured adaptive feedback indexes")
        except Exception as exc:
            logger.warning("Could not ensure adaptive feedback indexes: %s", exc)

    # ------------------------------------------------------------------
    # Evaluate a completed gap analysis run
    # ------------------------------------------------------------------

    async def evaluate(self, run_result: GapAnalysisResponse, run_id: str) -> Optional[str]:
        """Call the critic LLM to evaluate a gap analysis run and persist feedback.

        Returns the ``_id`` of the new feedback document, or None on failure.
        """
        if not self._cfg.adaptive_feedback_enabled:
            return None

        try:
            summary_json = json.dumps(run_result.summary.model_dump(), indent=2)
            gaps_json = json.dumps(
                [g.model_dump(exclude={"policy_subchunk_text", "policy_combined_sections", "statute_subchunk_text", "statute_chunk_text"})
                 for g in run_result.gaps],
                indent=2,
                default=str,
            )

            from prompt_loader import load_prompt_yaml, render_prompt

            cfg = load_prompt_yaml(self._cfg.adaptive_critic_prompt_path)
            prompt = render_prompt(
                cfg["prompt"],
                GAP_ANALYSIS_RESULTS=gaps_json,
                SUMMARY=summary_json,
            )

            critic_output = await self._llm._call_json(prompt, max_tokens=2048)
            if not critic_output:
                logger.warning("Critic LLM returned no output for run %s", run_id)
                return None

            suggestions = self._validate_suggestions(critic_output.get("suggestions", []))
            summary_evaluation = (critic_output.get("summary_evaluation") or "").strip()

            policy_id = run_result.policy_document_id
            policy_ids = run_result.policy_document_ids

            feedback_id = str(uuid.uuid4())
            doc: Dict[str, Any] = {
                "_id": feedback_id,
                "run_id": run_id,
                "policy_document_id": policy_id,
                "policy_document_ids": policy_ids,
                "created_at": utc_now().isoformat(),
                "version": "v4",
                "summary_evaluation": summary_evaluation,
                "suggestions": suggestions,
                "gap_summary_snapshot": run_result.summary.model_dump(),
                "superseded_by": None,
            }

            def _write():
                coll = self._feedback_coll()
                coll.insert_one(doc)
                # Mark prior feedback for this policy as superseded
                coll.update_many(
                    {
                        "policy_document_id": policy_id,
                        "superseded_by": None,
                        "_id": {"$ne": feedback_id},
                    },
                    {"$set": {"superseded_by": feedback_id}},
                )

            await run_in_thread(_write)
            logger.info(
                "Critic evaluation stored for run %s (%d suggestions)",
                run_id, len(suggestions),
            )
            return feedback_id

        except Exception as exc:
            logger.warning("Critic evaluation failed for run %s: %s", run_id, exc)
            return None

    # ------------------------------------------------------------------
    # Retrieve active feedback for injection into the next run
    # ------------------------------------------------------------------

    def get_active_feedback(
        self, policy_document_id: str, limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent non-superseded feedback for a policy."""
        max_items = limit or self._cfg.adaptive_feedback_max_items
        docs = list(
            self._feedback_coll()
            .find({"policy_document_id": policy_document_id, "superseded_by": None})
            .sort("created_at", -1)
            .limit(max_items)
        )
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return docs

    def format_feedback_for_prompt(self, feedback_docs: List[Dict[str, Any]]) -> str:
        """Render feedback documents into a text block for the LLM prompt."""
        instructions: List[str] = []
        for doc in feedback_docs:
            for suggestion in doc.get("suggestions", []):
                instruction = (suggestion.get("instruction") or "").strip()
                if instruction:
                    instructions.append(f"- {instruction}")
        if not instructions:
            return ""
        header = "PRIOR ANALYSIS FEEDBACK (incorporate these lessons into your analysis):\n\n"
        return header + "\n".join(instructions)

    # ------------------------------------------------------------------
    # Record which feedback was consumed by a run
    # ------------------------------------------------------------------

    async def record_feedback_usage(
        self,
        run_id: str,
        feedback_ids: List[str],
        rendered_text: str,
    ) -> None:
        """Append a record to adaptive_feedback_log."""
        doc = {
            "_id": str(uuid.uuid4()),
            "run_id": run_id,
            "feedback_ids_used": feedback_ids,
            "feedback_instructions_text": rendered_text,
            "created_at": utc_now().isoformat(),
        }
        await run_in_thread(lambda: self._log_coll().insert_one(doc))

    # ------------------------------------------------------------------
    # Audit trail queries
    # ------------------------------------------------------------------

    def get_feedback_history(
        self,
        policy_document_id: str,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return chronological feedback history for a policy."""
        query: Dict[str, Any] = {"policy_document_id": policy_document_id}
        if active_only:
            query["superseded_by"] = None
        docs = list(
            self._feedback_coll()
            .find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return docs

    def get_feedback_log(
        self,
        run_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return feedback consumption log entries."""
        query: Dict[str, Any] = {}
        if run_id:
            query["run_id"] = run_id
        docs = list(
            self._log_coll()
            .find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return docs

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_suggestions(raw: Any) -> List[Dict[str, Any]]:
        """Sanitize and validate the suggestions array from the critic LLM."""
        if not isinstance(raw, list):
            return []
        validated: List[Dict[str, Any]] = []
        for item in raw[:10]:
            if not isinstance(item, dict):
                continue
            category = (item.get("category") or "").strip().lower()
            if category not in VALID_CATEGORIES:
                continue
            severity = (item.get("severity") or "medium").strip().lower()
            if severity not in VALID_SEVERITIES:
                severity = "medium"
            instruction = (item.get("instruction") or "").strip()
            if not instruction:
                continue
            validated.append({
                "category": category,
                "description": (item.get("description") or "").strip(),
                "instruction": instruction,
                "statute_reference": item.get("statute_reference") or None,
                "severity": severity,
            })
        return validated
