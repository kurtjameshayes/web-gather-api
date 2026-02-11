"""Pydantic DTOs for policy statute compliance API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ConfidenceThresholds(BaseModel):
    compliant: float = Field(default=0.75, ge=0.0, le=1.0)
    non_compliant: float = Field(default=0.75, ge=0.0, le=1.0)


class RequestOptions(BaseModel):
    explainability: bool = True
    redact_pii: bool = True


class PolicyStatuteComplianceRequest(BaseModel):
    policy_collection: str = Field(..., min_length=1)
    policy_id: str = Field(..., min_length=1)
    jurisdiction: str = Field(..., min_length=1)


class AppliedStatute(BaseModel):
    statute_id: str
    jurisdiction: str
    title: str
    section_id: str
    matched_span: str
    evidence_score: float


class PolicySectionResult(BaseModel):
    section_id: str
    section_text: str
    applied_statutes: List[AppliedStatute]
    compliance: str
    confidence: float
    rationale: str
    remediation_suggestions: List[str]
    retrieval_trace: List[str]


class SummaryCounts(BaseModel):
    compliant: int
    non_compliant: int
    neither: int


class SummaryResult(BaseModel):
    overall_compliance: str
    counts: SummaryCounts


class PolicyStatuteComplianceResponse(BaseModel):
    policy_id: Optional[str]
    jurisdiction: str
    sections: List[PolicySectionResult]
    summary: SummaryResult
    warnings: List[str]
