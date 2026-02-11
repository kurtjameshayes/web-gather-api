"""Compliance evaluation and post-processing logic."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from compliance_config import ConfidenceThresholds
from compliance_utils import clamp, extract_json_block
from vector_retriever import StatuteCandidate


ALLOWED_COMPLIANCE = {"compliant", "non_compliant", "neither"}


@dataclass
class EvaluatedSection:
    section_id: str
    section_text: str
    applied_statutes: List[Dict[str, Any]]
    compliance: str
    confidence: float
    rationale: str
    remediation_suggestions: List[str]
    retrieval_trace: List[str]
    warnings: List[str]


class ComplianceEvaluator:
    def __init__(self, evidence_score_threshold: float) -> None:
        self._evidence_score_threshold = evidence_score_threshold

    def evaluate(
        self,
        section_id: str,
        section_text: str,
        candidates: List[StatuteCandidate],
        llm_response_text: str,
        thresholds: ConfidenceThresholds,
        explainability: bool,
    ) -> EvaluatedSection:
        warnings: List[str] = []
        llm_payload = self._parse_llm_json(llm_response_text, warnings)

        applied_statutes = self._build_applied_statutes(llm_payload, candidates)
        retrieval_trace = (
            [f"{c.statute_id}:{c.chunk_id}:{c.score:.4f}" for c in candidates]
            if explainability
            else []
        )

        compliance_raw = (llm_payload.get("compliance") or "neither").strip().lower()
        if compliance_raw not in ALLOWED_COMPLIANCE:
            warnings.append("LLM returned invalid compliance value; defaulted to neither.")
            compliance_raw = "neither"

        confidence = clamp(float(llm_payload.get("confidence", 0.0)))
        compliance = self._map_compliance(compliance_raw, confidence, thresholds)

        rationale = (llm_payload.get("rationale") or "").strip()
        if not rationale:
            rationale = "Insufficient evidence to determine compliance with confidence."

        remediation = llm_payload.get("remediation_suggestions") or []
        remediation = [str(item).strip() for item in remediation if str(item).strip()]
        if not remediation:
            remediation = ["Clarify policy language and align with statutory requirements."]

        compliance, warnings = self._apply_rule_checks(
            compliance, section_text, candidates, warnings
        )

        max_evidence = max((item.get("evidence_score", 0.0) for item in applied_statutes), default=0.0)
        if candidates and max_evidence < self._evidence_score_threshold:
            warnings.append("Low retrieval evidence score; marked as neither.")
            compliance = "neither"

        return EvaluatedSection(
            section_id=section_id,
            section_text=section_text,
            applied_statutes=applied_statutes,
            compliance=compliance,
            confidence=confidence,
            rationale=rationale,
            remediation_suggestions=remediation,
            retrieval_trace=retrieval_trace,
            warnings=warnings,
        )

    def _parse_llm_json(self, text: str, warnings: List[str]) -> Dict[str, Any]:
        if not text:
            warnings.append("LLM response was empty.")
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_block = extract_json_block(text)
            if json_block:
                try:
                    return json.loads(json_block)
                except json.JSONDecodeError:
                    warnings.append("Failed to parse LLM JSON response.")
            else:
                warnings.append("No JSON object found in LLM response.")
        return {}

    def _build_applied_statutes(
        self, llm_payload: Dict[str, Any], candidates: List[StatuteCandidate]
    ) -> List[Dict[str, Any]]:
        applied = llm_payload.get("applied_statutes") or []
        applied_list = []
        candidate_map = {c.statute_id: c for c in candidates}
        for item in applied:
            statute_id = str(item.get("statute_id", "")).strip()
            candidate = candidate_map.get(statute_id)
            applied_list.append(
                {
                    "statute_id": statute_id or (candidate.statute_id if candidate else ""),
                    "jurisdiction": str(item.get("jurisdiction") or (candidate.jurisdiction if candidate else "")),
                    "title": str(item.get("title") or (candidate.title if candidate else "")),
                    "section_id": str(item.get("section_id") or (candidate.section_id if candidate else "")),
                    "matched_span": str(item.get("matched_span") or ""),
                    "evidence_score": float(item.get("evidence_score") or (candidate.score if candidate else 0.0)),
                }
            )

        if not applied_list:
            for candidate in candidates:
                applied_list.append(
                    {
                        "statute_id": candidate.statute_id,
                        "jurisdiction": candidate.jurisdiction,
                        "title": candidate.title,
                        "section_id": candidate.section_id,
                        "matched_span": "",
                        "evidence_score": candidate.score,
                    }
                )
        return applied_list

    def _map_compliance(
        self, compliance_raw: str, confidence: float, thresholds: ConfidenceThresholds
    ) -> str:
        # Thresholds gate the final status to avoid overconfident labels.
        if compliance_raw == "compliant" and confidence >= thresholds.compliant:
            return "compliant"
        if compliance_raw == "non_compliant" and confidence >= thresholds.non_compliant:
            return "non_compliant"
        return "neither"

    def _apply_rule_checks(
        self,
        compliance: str,
        section_text: str,
        candidates: List[StatuteCandidate],
        warnings: List[str],
    ) -> Tuple[str, List[str]]:
        if compliance != "compliant":
            return compliance, warnings

        statute_requires_consent = any(
            "consent" in (candidate.chunk_text or "").lower()
            or "opt-in" in (candidate.chunk_text or "").lower()
            for candidate in candidates
        )
        policy_mentions_consent = "consent" in section_text.lower() or "opt-in" in section_text.lower()

        if statute_requires_consent and not policy_mentions_consent:
            warnings.append("Consent requirement detected without corresponding policy language.")
            return "non_compliant", warnings

        return compliance, warnings
