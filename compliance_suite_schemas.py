"""Pydantic request/response models for the compliance suite API (gap analysis, health score, drift, etc.)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ----- Applicability (6.1) -----


class ApplicabilityRequest(BaseModel):
    policy_document_id: Optional[str] = None
    text: Optional[str] = None
    database: Optional[str] = None
    policy_collection: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self) -> "ApplicabilityRequest":
        if not self.policy_document_id and not self.text:
            raise ValueError("Either policy_document_id or text must be provided.")
        return self


class ApplicabilityResponse(BaseModel):
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    confidence: Optional[Dict[str, float]] = None


# ----- Gap Analysis (2.x) -----


class GapItem(BaseModel):
    jurisdiction: str
    statute_reference: str
    statute_name: Optional[str] = None
    statute_chunk_id: Optional[str] = None
    section: Optional[str] = None
    requirement_summary: str
    status: str = Field(..., pattern="^(missing|addressed|conflict|partial|ambiguous)$")
    policy_quote: Optional[str] = None
    statute_quote: Optional[str] = None
    conflict_description: Optional[str] = None
    analysis_failed: bool = False
    confidence: Optional[str] = Field(None, pattern="^(high|medium|low)$")  # v3
    citation_binding_failed: Optional[bool] = None  # v3: True when policy_quote not in policy text
    # Subchunk gap analysis context
    policy_subchunk_text: Optional[str] = None
    policy_combined_sections: Optional[str] = None  # Combined policy text compared (v3: top-k matches with context)
    statute_subchunk_text: Optional[str] = None
    statute_chunk_text: Optional[str] = None


class GapSummary(BaseModel):
    total_requirements: int = 0
    missing: int = 0
    addressed: int = 0
    conflicts: int = 0
    partial: int = 0
    ambiguous: int = 0
    analysis_failures: int = 0  # v3: count of items where analysis_failed


class RetrievalMetadata(BaseModel):
    """Metadata about statute-policy retrieval for transparency when gaps=[]."""

    statute_subchunks_considered: int = 0
    statute_chunks_considered: int = 0  # v2: chunk-level
    statute_items_considered: int = 0  # v3: generic term
    statute_pairs_matched: int = 0


class GapAnalysisRequest(BaseModel):
    policy_document_id: Optional[str] = None
    policy_document_ids: Optional[List[str]] = None  # v3: list of policy IDs, treated as single combined document for vector search
    company_name: Optional[str] = None
    applicable_jurisdictions: Optional[List[str]] = None
    statute_document_id: Optional[str] = None  # Optional filter for statute subchunks
    database: Optional[str] = None
    policy_collection: Optional[str] = None
    save_results: bool = True  # If False, do not persist (e.g. for report run_now)
    num_rows: Optional[int] = Field(None, gt=0)  # If set, limit to this many statute subchunks (partial run)
    run_async: bool = True  # If True (default), start background job and return job_id immediately

    @model_validator(mode="after")
    def validate_policy_source(self) -> "GapAnalysisRequest":
        has_single = bool(self.policy_document_id and self.policy_document_id.strip())
        has_list = bool(self.policy_document_ids and len(self.policy_document_ids) > 0)
        if not has_single and not has_list:
            raise ValueError("Either policy_document_id or policy_document_ids (non-empty) must be provided.")
        return self


class GapAnalysisResponse(BaseModel):
    policy_document_id: str
    policy_document_ids: Optional[List[str]] = None  # Populated when request used policy_document_ids
    company_name: Optional[str] = None
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str  # ISO8601
    gaps: List[GapItem] = Field(default_factory=list)
    summary: GapSummary = Field(default_factory=GapSummary)
    retrieval_metadata: Optional[RetrievalMetadata] = None
    # For job path: used when writing compliance_results/run_log after action completes
    statute_chunk_ids_used: Optional[List[str]] = None
    run_types: Optional[List[str]] = None  # e.g. ["gap"], ["gap_v2"], ["gap_v3"]
    run_type: Optional[str] = None  # v3: "gap_analysis_v3"
    version: Optional[str] = None  # v3: "v3"


# ----- Multi-Jurisdictional (3.x) -----


class StrictestDenominatorItem(BaseModel):
    canonical_requirement_id: str
    label: str
    strictest_jurisdiction: str
    strictest_description: str
    all_jurisdictions: List[str] = Field(default_factory=list)
    policy_alignment: str = Field(
        default="not_provided",
        pattern="^(satisfies_all|satisfies_strictest_only|conflict_between_jurisdictions|not_provided|unclear)$",
    )
    policy_note: Optional[str] = None


class ConflictBetweenJurisdictionsItem(BaseModel):
    canonical_requirement_id: str
    jurisdiction_a: str
    jurisdiction_b: str
    conflict_summary: str


class MultiJurisdictionalRequest(BaseModel):
    applicable_jurisdictions: List[str] = Field(..., min_length=1)
    policy_document_id: Optional[str] = None
    database: Optional[str] = None
    policy_collection: Optional[str] = None


class MultiJurisdictionalResponse(BaseModel):
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str  # ISO8601
    strictest_common_denominator: List[StrictestDenominatorItem] = Field(default_factory=list)
    conflicts_between_jurisdictions: List[ConflictBetweenJurisdictionsItem] = Field(default_factory=list)


# ----- Health Score (4.x) -----


class HealthScoreRequest(BaseModel):
    policy_document_id: str = Field(..., min_length=1)
    applicable_jurisdictions: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    database: Optional[str] = None
    policy_collection: Optional[str] = None
    save_results: bool = True  # If False, do not persist (e.g. for report run_now)
    run_async: bool = True  # If True (default), start background job and return job_id immediately


def _score_assessment(
    score: Optional[int],
    error: Optional[str],
    components: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a multi-sentence assessment with reasoning and improvement suggestions."""
    if error or score is None:
        return (
            "Unable to assess the privacy health score. "
            "This may occur when no applicable statutes were found, analysis could not be completed, "
            "or the policy document was not sufficiently indexed. "
            "Ensure the policy is indexed in policy_legal_embeddings and that statute mappings exist for your jurisdictions."
        )
    total = (components or {}).get("requirements_total") or 0
    addressed = (components or {}).get("addressed") or 0
    missing = (components or {}).get("missing") or 0
    conflicts = (components or {}).get("conflicts") or 0
    partial = (components or {}).get("partial", 0)
    ambiguous = (components or {}).get("ambiguous", 0)

    if score >= 90:
        level = "Excellent"
        reasoning = (
            f"Your policy fully addresses {addressed} of {total} applicable privacy requirements. "
            if total else "Your policy demonstrates strong alignment with applicable privacy statutes. "
        )
        suggestions = (
            "Maintain this level by keeping disclosures current as regulations evolve and reviewing new jurisdictional requirements."
        )
    elif score >= 75:
        level = "Good"
        reasoning = (
            f"Your policy addresses {addressed} of {total} requirements, with {partial} partially addressed. "
            if total else "Your policy shows solid coverage of key privacy obligations. "
        )
        suggestions = (
            "To reach an excellent score, strengthen partially addressed requirements with clearer, more explicit language "
            "and ensure any gaps are closed. Review the gap analysis for specific requirements to improve."
        )
    elif score >= 50:
        level = "Fair"
        reasoning = (
            f"Your policy addresses {addressed} of {total} requirements, with {partial} partially addressed "
            f"and {missing} missing. "
            if total else "Your policy covers some privacy requirements but has notable gaps. "
        )
        suggestions = (
            "To improve, add explicit disclosures for missing requirements and strengthen partial coverage. "
            "Prioritize consumer rights (e.g., right to know, delete, opt-out) and controller duties (e.g., data minimization, security). "
            "Review the gap analysis output for specific statute references and recommended language."
        )
    elif score >= 25:
        level = "Needs improvement"
        reasoning = (
            f"Your policy addresses {addressed} of {total} requirements, with {missing} missing and {partial} partially addressed. "
            if total else "Your policy has significant gaps relative to applicable privacy statutes. "
        )
        suggestions = (
            "Add clear, binding language for missing consumer rights and controller duties. "
            "Ensure opt-out mechanisms, data retention limits, and security obligations are explicitly stated. "
            "Consider a comprehensive privacy policy review against each applicable jurisdiction's requirements."
        )
    else:
        level = "Critical"
        reasoning = (
            f"Your policy addresses {addressed} of {total} requirements, with {missing} missing. "
            if total else "Your policy has critical gaps and may not meet baseline privacy compliance expectations. "
        )
        suggestions = (
            "Immediate action is recommended. Add explicit disclosures for all applicable consumer rights "
            "(right to know, access, delete, correct, opt-out of sale/sharing) and controller duties. "
            "Ensure the policy is comprehensive, unambiguous, and aligned with each jurisdiction's statutory language. "
            "A full gap analysis and legal review is strongly advised."
        )

    if conflicts > 0:
        conflict_note = (
            f" There are {conflicts} requirement(s) where policy language conflicts with statute expectations, "
            "which reduces the score. Resolve these conflicts by aligning policy language with statutory requirements."
        )
    else:
        conflict_note = ""

    if ambiguous > 0 and level not in ("Excellent", "Good"):
        ambiguous_note = (
            f" {ambiguous} requirement(s) were marked ambiguous; clarifying this language could improve the score."
        )
    else:
        ambiguous_note = ""

    return f"{level}. {reasoning}{suggestions}{conflict_note}{ambiguous_note}"


class HealthScoreResponse(BaseModel):
    policy_document_id: str
    company_name: Optional[str] = None
    privacy_health_score: Optional[int] = None  # 0-100 or null if error
    score_assessment: Optional[str] = None  # Human-readable assessment (e.g. "Good", "Fair")
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)  # by_jurisdiction, by_category
    components: Dict[str, Any] = Field(default_factory=dict)
    analyzed_at: str  # ISO8601
    error: Optional[str] = None  # e.g. "no_applicable_statutes", "insufficient_analysis"


# ----- Drift (5.x) -----


class DriftGapItem(BaseModel):
    jurisdiction: str
    requirement_summary: str
    statute_reference: str


class DriftAlertItem(BaseModel):
    alert_id: str
    type: str = "regulatory_drift"
    policy_document_id: str
    company_name: Optional[str] = None
    trigger: str
    affected_jurisdictions: List[str] = Field(default_factory=list)
    new_gaps: List[DriftGapItem] = Field(default_factory=list)
    resolved_gaps: List[DriftGapItem] = Field(default_factory=list)
    score_delta: Optional[int] = None
    previous_score: Optional[int] = None
    current_score: Optional[int] = None
    detected_at: str  # ISO8601


class DriftCheckRequest(BaseModel):
    since: Optional[str] = None  # ISO8601 timestamp
    policy_document_ids: Optional[List[str]] = None
    full_rebaseline: bool = False


class DriftCheckResponse(BaseModel):
    alerts: List[DriftAlertItem] = Field(default_factory=list)
    policies_checked: int = 0
    alerts_written: int = 0


# ----- Report (new spec) -----


class ReportRequest(BaseModel):
    policy_document_id: str = Field(..., min_length=1)
    format: str = Field(..., pattern="^(markdown|pdf)$")
    source: str = Field(default="latest_stored", pattern="^(latest_stored|run_now)$")
    applicable_jurisdictions: Optional[List[str]] = None
    include_gap: bool = True
    include_health_score: bool = True
    include_multi_jurisdictional: bool = False


# ----- Runs list/detail (new spec) -----


class RunSummaryItem(BaseModel):
    run_id: str
    policy_document_id: str
    company_name: Optional[str] = None
    run_at: str  # ISO8601
    types: List[str] = Field(default_factory=list)
    privacy_health_score: Optional[int] = None
    score_assessment: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


class RunsListResponse(BaseModel):
    runs: List[RunSummaryItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


# ----- Citations (new spec) -----


class CitationItem(BaseModel):
    policy_excerpt: str
    policy_chunk_id: Optional[str] = None
    statute_reference: str
    jurisdiction: str
    alignment: bool
    statute_excerpt: Optional[str] = None


class CitationsRequest(BaseModel):
    policy_document_id: str = Field(..., min_length=1)
    applicable_jurisdictions: Optional[List[str]] = None


class CitationsSummary(BaseModel):
    total_citations: int = 0
    aligned: int = 0
    not_aligned: int = 0


class CitationsResponse(BaseModel):
    policy_document_id: str
    company_name: Optional[str] = None
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str  # ISO8601
    citations: List[CitationItem] = Field(default_factory=list)
    summary: CitationsSummary = Field(default_factory=CitationsSummary)


# ----- Risk assessment (new spec) -----


class RiskAssessmentRequest(BaseModel):
    policy_document_id: str = Field(..., min_length=1)
    applicable_jurisdictions: Optional[List[str]] = None
    template_id: Optional[str] = None
    include_report: bool = False


class RiskAssessmentResponse(BaseModel):
    policy_document_id: str
    company_name: Optional[str] = None
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    template_id: str = "default"
    analyzed_at: str  # ISO8601
    assessment: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[str] = None


class TemplateItem(BaseModel):
    id: str
    label: str


class TemplatesResponse(BaseModel):
    templates: List[TemplateItem] = Field(default_factory=list)


# ----- Alerts list (new spec; response shape) -----


class AlertListItem(BaseModel):
    alert_id: str
    type: str = "regulatory_drift"
    policy_document_id: str
    company_name: Optional[str] = None
    trigger: str
    affected_jurisdictions: List[str] = Field(default_factory=list)
    new_gaps: List[Any] = Field(default_factory=list)
    resolved_gaps: List[Any] = Field(default_factory=list)
    score_delta: Optional[int] = None
    previous_score: Optional[int] = None
    current_score: Optional[int] = None
    detected_at: str  # ISO8601


class AlertsListResponse(BaseModel):
    alerts: List[AlertListItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


# ----- Suggest Policy (policy rewrite from gap analysis) -----


class SuggestPolicyRequest(BaseModel):
    policy_text: str = Field(..., min_length=1)
    gap_analysis_text: str = Field(..., min_length=1)
    gap_analysis_match: str = Field(..., min_length=1)
    statute_text: str = Field(..., min_length=1)


class SuggestPolicyResponse(BaseModel):
    suggested_policy_text: str
    modifications_description: str
    analyzed_at: str  # ISO8601


# ----- Consumer Rights Request Router -----

VALID_REQUEST_TYPES = ("deletion", "access", "optout", "correction")

REQUEST_TYPE_LABELS: Dict[str, str] = {
    "deletion": "Right to Delete",
    "access": "Right to Access / Know",
    "optout": "Right to Opt-Out of Sale",
    "correction": "Right to Correction",
}


class DecisionTreeNode(BaseModel):
    """Recursive decision-tree node. Either a question (branch) or action (leaf)."""

    id: str
    question: Optional[str] = None
    yes: Optional["DecisionTreeNode"] = None
    no: Optional["DecisionTreeNode"] = None
    action: Optional[str] = None
    detail: Optional[str] = None
    sla: Optional[str] = None
    exceptions: Optional[List[str]] = None


DecisionTreeNode.model_rebuild()


class PolicyGapItem(BaseModel):
    covered: bool
    gap: Optional[str] = None


class JurisdictionInfo(BaseModel):
    name: str
    abbr: str


class ConsumerRightsRouterRequest(BaseModel):
    policy_document_id: Optional[str] = None
    text: Optional[str] = None
    applicable_jurisdictions: Optional[List[str]] = None
    request_types: Optional[List[str]] = None
    save_results: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> "ConsumerRightsRouterRequest":
        if not self.policy_document_id and not self.text:
            raise ValueError("Either policy_document_id or text must be provided.")
        return self

    @model_validator(mode="after")
    def validate_request_types(self) -> "ConsumerRightsRouterRequest":
        if self.request_types:
            invalid = [rt for rt in self.request_types if rt not in VALID_REQUEST_TYPES]
            if invalid:
                raise ValueError(f"Invalid request_types: {invalid}. Must be one of {VALID_REQUEST_TYPES}.")
        return self


class ConsumerRightsRouterResponse(BaseModel):
    policy_document_id: Optional[str] = None
    company_name: Optional[str] = None
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str  # ISO8601
    states: Dict[str, JurisdictionInfo] = Field(default_factory=dict)
    request_types: Dict[str, str] = Field(default_factory=dict)
    trees: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    policy_gaps: Dict[str, Dict[str, PolicyGapItem]] = Field(default_factory=dict)
