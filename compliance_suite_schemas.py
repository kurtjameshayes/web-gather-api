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
    company_name: Optional[str] = None
    applicable_jurisdictions: Optional[List[str]] = None
    statute_document_id: Optional[str] = None  # Optional filter for statute subchunks
    database: Optional[str] = None
    policy_collection: Optional[str] = None
    save_results: bool = True  # If False, do not persist (e.g. for report run_now)
    num_rows: Optional[int] = Field(None, gt=0)  # If set, limit to this many statute subchunks (partial run)
    run_async: bool = True  # If True (default), start background job and return job_id immediately


class GapAnalysisResponse(BaseModel):
    policy_document_id: str
    company_name: Optional[str] = None
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str  # ISO8601
    gaps: List[GapItem] = Field(default_factory=list)
    summary: GapSummary = Field(default_factory=GapSummary)
    retrieval_metadata: Optional[RetrievalMetadata] = None


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


class HealthScoreResponse(BaseModel):
    policy_document_id: str
    company_name: Optional[str] = None
    privacy_health_score: Optional[int] = None  # 0-100 or null if error
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
