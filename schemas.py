"""Pydantic DTOs for statute-policy compliance API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from compliance_suite_schemas import GapItem, GapSummary, RetrievalMetadata


class ConfidenceThresholds(BaseModel):
    compliant: float = Field(default=0.75, ge=0.0, le=1.0)
    non_compliant: float = Field(default=0.75, ge=0.0, le=1.0)


class RequestOptions(BaseModel):
    explainability: bool = True
    redact_pii: bool = True


class StatutePolicyComplianceRequest(BaseModel):
    policy_id: str = Field(..., min_length=1)
    jurisdiction: str = Field(..., min_length=1)
    policy_collection: Optional[str] = Field(default=None, min_length=1)


class StatutePolicyComplianceResponse(BaseModel):
    policy_id: str
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    analyzed_at: str
    gaps: List[GapItem] = Field(default_factory=list)
    summary: GapSummary = Field(default_factory=GapSummary)
    retrieval_metadata: Optional[RetrievalMetadata] = None
    warnings: List[str] = Field(default_factory=list)
