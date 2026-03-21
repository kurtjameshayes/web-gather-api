"""Endpoint routes and OpenAPI specification for the Web Gather API."""
from __future__ import annotations

from flask import Blueprint, Response, redirect, jsonify

routes_bp = Blueprint("routes", __name__)

# Must match app.py url_prefix when registering compliance_bp.
COMPLIANCE_API_PREFIX = "/api/compliance"
COMPLIANCE_V2_API_PREFIX = "/api/v2/compliance"
COMPLIANCE_V3_API_PREFIX = "/api/v3/compliance"
COMPLIANCE_V4_API_PREFIX = "/api/v4/compliance"


def build_openapi_spec():
    """Build and return the OpenAPI specification."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Web Gather API",
            "version": "1.0.0",
            "description": (
                "Gather, load, index, and search web documents. The /ingest "
                "endpoint loads documents only - use /create-embeddings separately "
                "to create vector embeddings."
            ),
        },
        "servers": [{"url": "/", "description": "API root (relative to current host)"}],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-api-key",
                    "description": "API key required when APP_API_KEY is configured (core/db/util) or COMPLIANCE_API_KEY is set (compliance).",
                },
            },
            "schemas": {
                "ConfidenceThresholds": {
                    "type": "object",
                    "description": "Minimum confidence scores (0-1) for compliant and non_compliant determinations.",
                    "properties": {
                        "compliant": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.75,
                            "description": "Lower bound for a compliant determination (0-1).",
                        },
                        "non_compliant": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.75,
                            "description": "Lower bound for a non-compliant determination (0-1).",
                        },
                    },
                },
                "ComplianceOptions": {
                    "type": "object",
                    "description": "Request options for explainability and PII redaction.",
                    "properties": {
                        "explainability": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include rationale and remediation suggestions in the response.",
                        },
                        "redact_pii": {
                            "type": "boolean",
                            "default": True,
                            "description": "Redact PII in returned text.",
                        },
                    },
                },
                "StatutePolicyComplianceRequest": {
                    "type": "object",
                    "description": "Request body for statute-policy compliance. Iterates statutory requirements via category mappings and finds matching policy chunks to evaluate gaps.",
                    "properties": {
                        "policy_id": {
                            "type": "string",
                            "description": "Document ID of the policy (document_id in the policies collection).",
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "Jurisdiction for statute comparison (e.g. CA, VA, CCPA).",
                        },
                        "policy_collection": {
                            "type": "string",
                            "description": "Optional collection name for indexed policy chunks. Defaults to policy_legal_embeddings when omitted.",
                        },
                    },
                    "required": ["policy_id", "jurisdiction"],
                },
                "GapItemSchema": {
                    "type": "object",
                    "description": "A single statutory requirement gap item.",
                    "properties": {
                        "jurisdiction": {"type": "string"},
                        "statute_reference": {"type": "string"},
                        "requirement_summary": {"type": "string"},
                        "status": {"type": "string", "enum": ["missing", "addressed", "conflict", "partial", "ambiguous"]},
                        "policy_quote": {"type": "string", "nullable": True},
                        "statute_quote": {"type": "string", "nullable": True},
                        "conflict_description": {"type": "string", "nullable": True},
                        "analysis_failed": {"type": "boolean"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"], "nullable": True},
                    },
                    "required": ["jurisdiction", "statute_reference", "requirement_summary", "status"],
                },
                "GapSummarySchema": {
                    "type": "object",
                    "description": "Summary counts for gap analysis results.",
                    "properties": {
                        "total_requirements": {"type": "integer"},
                        "missing": {"type": "integer"},
                        "addressed": {"type": "integer"},
                        "conflicts": {"type": "integer"},
                        "partial": {"type": "integer"},
                        "ambiguous": {"type": "integer"},
                        "analysis_failures": {"type": "integer"},
                    },
                },
                "StatutePolicyComplianceResponse": {
                    "type": "object",
                    "properties": {
                        "policy_id": {"type": "string"},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "analyzed_at": {"type": "string", "description": "ISO8601 timestamp"},
                        "gaps": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/GapItemSchema"},
                        },
                        "summary": {"$ref": "#/components/schemas/GapSummarySchema"},
                        "retrieval_metadata": {"type": "object", "nullable": True},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["policy_id", "applicable_jurisdictions", "analyzed_at", "gaps", "summary"],
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                },
                "ApplicabilityRequest": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "text": {"type": "string"},
                        "database": {"type": "string"},
                        "policy_collection": {"type": "string"},
                    },
                },
                "ApplicabilityResponse": {
                    "type": "object",
                    "properties": {
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "object", "additionalProperties": {"type": "number"}},
                    },
                },
                "GapItem": {
                    "type": "object",
                    "properties": {
                        "jurisdiction": {"type": "string"},
                        "statute_reference": {"type": "string"},
                        "statute_name": {"type": "string", "nullable": True},
                        "statute_chunk_id": {"type": "string", "nullable": True},
                        "section": {"type": "string", "nullable": True},
                        "requirement_summary": {"type": "string"},
                        "status": {"type": "string", "enum": ["missing", "addressed", "conflict", "partial", "ambiguous"]},
                        "policy_quote": {"type": "string", "nullable": True},
                        "statute_quote": {"type": "string", "nullable": True},
                        "conflict_description": {"type": "string", "nullable": True},
                        "analysis_failed": {"type": "boolean"},
                        "policy_subchunk_text": {"type": "string", "nullable": True, "description": "Policy subchunk text (subchunk gap analysis)"},
                        "policy_combined_sections": {"type": "string", "nullable": True, "description": "Combined policy text compared (v3: top-k matches with context; v1/v2: single chunk)"},
                        "statute_subchunk_text": {"type": "string", "nullable": True, "description": "Statute subchunk text (subchunk gap analysis)"},
                        "statute_chunk_text": {"type": "string", "nullable": True, "description": "Statute enclosing chunk (subchunk gap analysis)"},
                    },
                },
                "GapSummary": {
                    "type": "object",
                    "properties": {
                        "total_requirements": {"type": "integer"},
                        "missing": {"type": "integer"},
                        "addressed": {"type": "integer"},
                        "conflicts": {"type": "integer"},
                        "partial": {"type": "integer"},
                        "ambiguous": {"type": "integer"},
                    },
                },
                "GapAnalysisRequest": {
                    "type": "object",
                    "description": "Provide policy_document_id (single) or policy_document_ids (list). At least one required. v3: policy_document_ids treated as single combined document for vector search.",
                    "properties": {
                        "policy_document_id": {"type": "string", "description": "Single policy document ID."},
                        "policy_document_ids": {"type": "array", "items": {"type": "string"}, "description": "v3: List of policy document IDs, treated as single combined document for vector search."},
                        "company_name": {"type": "string", "nullable": True},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "statute_document_id": {"type": "string", "description": "Optional filter for statute subchunks"},
                        "database": {"type": "string"},
                        "policy_collection": {"type": "string"},
                        "save_results": {"type": "boolean", "default": True},
                        "num_rows": {"type": "integer", "minimum": 1, "description": "If set, limit to this many statute subchunks (partial run)"},
                        "run_async": {"type": "boolean", "default": True, "description": "If true (default), start background job and return job_id immediately (202). Use GET /jobs/{job_id} to poll. Set false for synchronous response."},
                    },
                },
                "ComplianceJobResponse": {
                    "type": "object",
                    "description": "Compliance background job status and result.",
                    "properties": {
                        "job_id": {"type": "string"},
                        "job_type": {"type": "string", "enum": ["gap_analysis", "health_score", "drift_check"]},
                        "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
                        "request": {"type": "object", "description": "Original request payload"},
                        "result": {"type": "object", "description": "Result when status=completed (e.g. GapAnalysisResponse)"},
                        "error": {"type": "string", "nullable": True, "description": "Error message when status=failed"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "started_at": {"type": "string", "format": "date-time", "nullable": True},
                        "completed_at": {"type": "string", "format": "date-time", "nullable": True},
                    },
                },
                "RetrievalMetadata": {
                    "type": "object",
                    "description": "Metadata about statute-policy retrieval for transparency when gaps=[].",
                    "properties": {
                        "statute_subchunks_considered": {"type": "integer", "description": "Number of statute subchunks retrieved for comparison (v1)."},
                        "statute_chunks_considered": {"type": "integer", "description": "Number of statute chunks retrieved for comparison (v2)."},
                        "statute_pairs_matched": {"type": "integer", "description": "Number of statute-policy pairs matched and analyzed."},
                    },
                },
                "GapAnalysisResponse": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "policy_document_ids": {"type": "array", "items": {"type": "string"}, "nullable": True, "description": "Populated when request used policy_document_ids (v3)."},
                        "company_name": {"type": "string", "nullable": True},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "analyzed_at": {"type": "string"},
                        "gaps": {"type": "array", "items": {"$ref": "#/components/schemas/GapItem"}},
                        "summary": {"$ref": "#/components/schemas/GapSummary"},
                        "retrieval_metadata": {"$ref": "#/components/schemas/RetrievalMetadata"},
                        "statute_chunk_ids_used": {"type": "array", "items": {"type": "string"}, "nullable": True, "description": "For job path: statute IDs used in run_log."},
                        "run_types": {"type": "array", "items": {"type": "string"}, "nullable": True, "description": "e.g. [\"gap\"], [\"gap_v2\"], [\"gap_v3\"]."},
                        "run_type": {"type": "string", "nullable": True, "description": "v3: gap_analysis_v3."},
                        "version": {"type": "string", "nullable": True, "description": "v3: v3."},
                    },
                },
                "HealthScoreRequest": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "weights": {"type": "object", "description": "Optional. Override default weights. Keys: exact match on requirement_summary[:50], or substring (e.g. 'right to know'). Default weights prioritize consumer rights (1.5), controller duties (1.2), processor duties (1.0)."},
                        "database": {"type": "string"},
                        "policy_collection": {"type": "string"},
                        "save_results": {"type": "boolean", "default": True},
                        "run_async": {"type": "boolean", "default": True, "description": "If true (default), start background job and return job_id immediately (202). Use GET /jobs/{job_id} to poll. Set false for synchronous response."},
                    },
                    "required": ["policy_document_id"],
                },
                "HealthScoreResponse": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "company_name": {"type": "string", "nullable": True},
                        "privacy_health_score": {"type": "integer", "nullable": True},
                        "score_assessment": {"type": "string", "nullable": True, "description": "Human-readable assessment (e.g. Excellent, Good, Fair, Needs improvement, Critical)"},
                        "score_breakdown": {"type": "object"},
                        "components": {"type": "object"},
                        "analyzed_at": {"type": "string"},
                        "error": {"type": "string", "nullable": True},
                    },
                },
                "MultiJurisdictionalRequest": {
                    "type": "object",
                    "properties": {
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "policy_document_id": {"type": "string"},
                        "database": {"type": "string"},
                        "policy_collection": {"type": "string"},
                    },
                    "required": ["applicable_jurisdictions"],
                },
                "MultiJurisdictionalResponse": {
                    "type": "object",
                    "properties": {
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "analyzed_at": {"type": "string"},
                        "strictest_common_denominator": {"type": "array"},
                        "conflicts_between_jurisdictions": {"type": "array"},
                    },
                },
                "DriftCheckRequest": {
                    "type": "object",
                    "properties": {
                        "since": {"type": "string", "description": "ISO8601 timestamp"},
                        "policy_document_ids": {"type": "array", "items": {"type": "string"}},
                        "full_rebaseline": {"type": "boolean", "default": False},
                    },
                },
                "DriftCheckResponse": {
                    "type": "object",
                    "properties": {
                        "alerts": {"type": "array"},
                        "policies_checked": {"type": "integer"},
                        "alerts_written": {"type": "integer"},
                    },
                },
                "GapCheckRequest": {
                    "type": "object",
                    "description": "Request body for a single statute-to-policy gap check. Both policy and statute are fetched from MongoDB.",
                    "properties": {
                        "policy_database_name": {
                            "type": "string",
                            "description": "Database containing the policy document",
                        },
                        "policy_collection_name": {
                            "type": "string",
                            "description": "Collection containing the policy document",
                        },
                        "policy_document_id": {
                            "type": "string",
                            "description": "Document ID of the policy",
                        },
                        "statute_database_name": {
                            "type": "string",
                            "description": "Database containing the statute requirement document",
                        },
                        "statute_collection_name": {
                            "type": "string",
                            "description": "Collection containing the statute requirement document",
                        },
                        "statute_document_id": {
                            "type": "string",
                            "description": "Document ID of the statute requirement",
                        },
                        "max_policy_chars": {
                            "type": "integer",
                            "default": 8000,
                            "description": "Max characters of policy text to send to the LLM (truncated from start to keep end)",
                        },
                    },
                    "required": [
                        "policy_database_name",
                        "policy_collection_name",
                        "policy_document_id",
                        "statute_database_name",
                        "statute_collection_name",
                        "statute_document_id",
                    ],
                },
                "GapCheckResponse": {
                    "type": "object",
                    "properties": {
                        "gap_check": {
                            "type": "object",
                            "properties": {
                                "addressed": {"type": "boolean"},
                                "policy_quote": {"type": "string", "nullable": True},
                                "missing": {"type": "boolean"},
                                "conflict": {"type": "boolean"},
                                "conflict_description": {"type": "string", "nullable": True},
                            },
                            "required": ["addressed", "policy_quote", "missing", "conflict", "conflict_description"],
                        }
                    },
                    "required": ["gap_check"],
                },
                "ReportRequest": {
                    "type": "object",
                    "description": "Request body for generating a compliance report (Markdown or PDF).",
                    "properties": {
                        "policy_document_id": {"type": "string", "description": "Policy document ID."},
                        "format": {"type": "string", "enum": ["markdown", "pdf"], "description": "Report format."},
                        "source": {
                            "type": "string",
                            "enum": ["latest_stored", "run_now"],
                            "default": "latest_stored",
                            "description": "Use latest stored result or run gap+health now (no persist).",
                        },
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}, "description": "Used when source is run_now."},
                        "include_gap": {"type": "boolean", "default": True, "description": "Include gap analysis section."},
                        "include_health_score": {"type": "boolean", "default": True, "description": "Include Privacy Health Score."},
                        "include_multi_jurisdictional": {"type": "boolean", "default": False, "description": "Include strictest-common-denominator (run_now only)."},
                    },
                    "required": ["policy_document_id", "format"],
                },
                "ReportResponseMarkdown": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["markdown"]},
                        "content": {"type": "string", "description": "Markdown report content."},
                    },
                    "required": ["format", "content"],
                },
                "RunSummaryItem": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "policy_document_id": {"type": "string"},
                        "company_name": {"type": "string", "nullable": True},
                        "run_at": {"type": "string", "description": "ISO8601"},
                        "types": {"type": "array", "items": {"type": "string"}},
                        "privacy_health_score": {"type": "integer", "nullable": True},
                        "score_assessment": {"type": "string", "nullable": True},
                        "summary": {"type": "object", "properties": {"total_requirements": {"type": "integer"}, "missing": {"type": "integer"}, "addressed": {"type": "integer"}, "conflicts": {"type": "integer"}}},
                    },
                },
                "RunsListResponse": {
                    "type": "object",
                    "properties": {
                        "runs": {"type": "array", "items": {"$ref": "#/components/schemas/RunSummaryItem"}},
                        "total": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["runs", "total", "limit", "offset"],
                },
                "CitationsRequest": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["policy_document_id"],
                },
                "CitationItem": {
                    "type": "object",
                    "properties": {
                        "policy_excerpt": {"type": "string"},
                        "policy_chunk_id": {"type": "string", "nullable": True},
                        "statute_reference": {"type": "string"},
                        "jurisdiction": {"type": "string"},
                        "alignment": {"type": "boolean"},
                        "statute_excerpt": {"type": "string", "nullable": True},
                    },
                },
                "CitationsResponse": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "company_name": {"type": "string", "nullable": True},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "analyzed_at": {"type": "string"},
                        "citations": {"type": "array", "items": {"$ref": "#/components/schemas/CitationItem"}},
                        "summary": {"type": "object", "properties": {"total_citations": {"type": "integer"}, "aligned": {"type": "integer"}, "not_aligned": {"type": "integer"}}},
                    },
                },
                "RiskAssessmentRequest": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "template_id": {"type": "string", "nullable": True},
                        "include_report": {"type": "boolean", "default": False},
                    },
                    "required": ["policy_document_id"],
                },
                "RiskAssessmentResponse": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string"},
                        "company_name": {"type": "string", "nullable": True},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "template_id": {"type": "string"},
                        "analyzed_at": {"type": "string"},
                        "assessment": {"type": "object", "description": "processing_purposes, data_categories, risks, mitigations, gaps_from_statute"},
                        "report": {"type": "string", "nullable": True},
                    },
                },
                "TemplateItem": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
                },
                "TemplatesResponse": {
                    "type": "object",
                    "properties": {"templates": {"type": "array", "items": {"$ref": "#/components/schemas/TemplateItem"}}},
                },
                "AlertListItem": {
                    "type": "object",
                    "properties": {
                        "alert_id": {"type": "string"},
                        "type": {"type": "string"},
                        "policy_document_id": {"type": "string"},
                        "company_name": {"type": "string", "nullable": True},
                        "trigger": {"type": "string"},
                        "affected_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "new_gaps": {"type": "array"},
                        "resolved_gaps": {"type": "array"},
                        "score_delta": {"type": "integer", "nullable": True},
                        "previous_score": {"type": "integer", "nullable": True},
                        "current_score": {"type": "integer", "nullable": True},
                        "detected_at": {"type": "string"},
                    },
                },
                "AlertsListResponse": {
                    "type": "object",
                    "properties": {
                        "alerts": {"type": "array", "items": {"$ref": "#/components/schemas/AlertListItem"}},
                        "total": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["alerts", "total", "limit", "offset"],
                },
                "SuggestPolicyRequest": {
                    "type": "object",
                    "properties": {
                        "policy_text": {"type": "string", "description": "The current policy text to be revised for compliance."},
                        "gap_analysis_text": {"type": "string", "description": "The gap analysis finding describing the compliance gap."},
                        "gap_analysis_match": {"type": "string", "description": "The gap analysis match status (e.g. missing, conflict, partial)."},
                        "statute_text": {"type": "string", "description": "The authoritative statute text that the policy must comply with."},
                    },
                    "required": ["policy_text", "gap_analysis_text", "gap_analysis_match", "statute_text"],
                },
                "SuggestPolicyResponse": {
                    "type": "object",
                    "properties": {
                        "suggested_policy_text": {"type": "string", "description": "The full revised policy text with modifications applied."},
                        "modifications_description": {"type": "string", "description": "Plain-language summary of every change made and why."},
                        "analyzed_at": {"type": "string", "description": "ISO8601 timestamp of when the analysis was performed."},
                    },
                },
                "ConsumerRightsRouterRequest": {
                    "type": "object",
                    "description": "Request to generate consumer rights request handling decision trees and policy gap analysis.",
                    "properties": {
                        "policy_document_id": {"type": "string", "description": "Policy document ID in privacy-compliance.policy_legal_embeddings."},
                        "text": {"type": "string", "description": "Raw policy text (alternative to policy_document_id)."},
                        "applicable_jurisdictions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Jurisdictions to analyze (e.g. ['CA', 'VA', 'CO', 'TX']). Defaults to config.",
                        },
                        "request_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["deletion", "access", "optout", "correction"]},
                            "description": "Consumer rights request types to include. Defaults to all four.",
                        },
                        "save_results": {"type": "boolean", "default": True, "description": "Whether to persist results to compliance_results."},
                    },
                },
                "DecisionTreeNode": {
                    "type": "object",
                    "description": "Recursive decision tree node. Either a question (branch) with yes/no children, or an action (leaf) with operational details.",
                    "properties": {
                        "id": {"type": "string", "description": "Unique node identifier (e.g. 'ca-del-1')."},
                        "question": {"type": "string", "description": "Question text (present on branch nodes)."},
                        "yes": {"$ref": "#/components/schemas/DecisionTreeNode"},
                        "no": {"$ref": "#/components/schemas/DecisionTreeNode"},
                        "action": {"type": "string", "description": "Action title in caps (present on leaf nodes)."},
                        "detail": {"type": "string", "description": "Detailed operational instructions (leaf nodes)."},
                        "sla": {"type": "string", "description": "Statutory response deadline (leaf nodes)."},
                        "exceptions": {"type": "array", "items": {"type": "string"}, "description": "Applicable statutory exceptions (leaf nodes)."},
                    },
                    "required": ["id"],
                },
                "PolicyGapItem": {
                    "type": "object",
                    "properties": {
                        "covered": {"type": "boolean", "description": "Whether the policy adequately covers this right for this jurisdiction."},
                        "gap": {"type": "string", "nullable": True, "description": "Description of the policy gap with statute citation, or null if aligned."},
                    },
                    "required": ["covered"],
                },
                "JurisdictionInfo": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full jurisdiction name (e.g. 'California (CCPA/CPRA)')."},
                        "abbr": {"type": "string", "description": "Two-letter abbreviation (e.g. 'CA')."},
                    },
                    "required": ["name", "abbr"],
                },
                "ConsumerRightsRouterResponse": {
                    "type": "object",
                    "properties": {
                        "policy_document_id": {"type": "string", "nullable": True},
                        "company_name": {"type": "string", "nullable": True},
                        "applicable_jurisdictions": {"type": "array", "items": {"type": "string"}},
                        "analyzed_at": {"type": "string", "description": "ISO8601 timestamp."},
                        "states": {
                            "type": "object",
                            "description": "Map of state slug to JurisdictionInfo.",
                            "additionalProperties": {"$ref": "#/components/schemas/JurisdictionInfo"},
                        },
                        "request_types": {
                            "type": "object",
                            "description": "Map of request type key to display label.",
                            "additionalProperties": {"type": "string"},
                        },
                        "trees": {
                            "type": "object",
                            "description": "trees[request_type][state_slug] = DecisionTreeNode.",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": {"$ref": "#/components/schemas/DecisionTreeNode"},
                            },
                        },
                        "policy_gaps": {
                            "type": "object",
                            "description": "policy_gaps[request_type][state_slug] = PolicyGapItem.",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": {"$ref": "#/components/schemas/PolicyGapItem"},
                            },
                        },
                    },
                },
            }
        },
        "security": [{"ApiKeyAuth": []}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Returns service health status including database connectivity. No authentication required.",
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "ok"},
                                            "database": {"type": "string", "example": "ok"},
                                        },
                                    },
                                },
                            },
                        },
                        "503": {
                            "description": "Service is degraded",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "degraded"},
                                            "database": {"type": "string", "example": "error"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "tags": ["Health"],
                },
            },
            "/gather": {
                "post": {
                    "summary": "Gather web documents based on query",
                    "description": "Search the web for documents matching the query. Returns search results with URLs that can be selected for crawling via the /ingest endpoint. Each result includes a relevance score (0-1) and percent_match (0-100) indicating how well the content matches the query for AI consumption. The response includes a next_step object describing the required and optional parameters for ingestion.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results with next step instructions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string", "description": "The search query"},
                                            "results": {
                                                "type": "array",
                                                "description": "List of search results sorted by relevance",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "url": {"type": "string"},
                                                        "title": {"type": "string"},
                                                        "description": {"type": "string"},
                                                        "score": {
                                                            "type": "number",
                                                            "description": "Relevance score from Firecrawl search (0-1)",
                                                            "minimum": 0,
                                                            "maximum": 1,
                                                        },
                                                        "percent_match": {
                                                            "type": "number",
                                                            "description": "Relevance score as percentage (0-100)",
                                                            "minimum": 0,
                                                            "maximum": 100,
                                                        },
                                                    },
                                                },
                                            },
                                            "next_step": {
                                                "type": "object",
                                                "description": "Instructions for the next step: selecting a URL and calling /ingest",
                                                "properties": {
                                                    "action": {"type": "string"},
                                                    "endpoint": {"type": "string"},
                                                    "method": {"type": "string"},
                                                    "required_parameters": {
                                                        "type": "object",
                                                        "description": "Parameters that must be provided to /ingest",
                                                        "properties": {
                                                            "url": {"type": "object"},
                                                            "database": {"type": "object"},
                                                            "collection": {"type": "object"},
                                                        },
                                                    },
                                                    "optional_parameters": {
                                                        "type": "object",
                                                        "description": "Optional parameters for /ingest (depth, breadth, mode, index_database, index_collection)",
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/ingest": {
                "post": {
                    "summary": "Load a document by crawling a URL or parsing a PDF",
                    "description": (
                        "Supports both web pages (crawled via Firecrawl) and PDF "
                        "files (downloaded and parsed). PDF files are "
                        "automatically detected by URL extension or Content-Type "
                        "header. Use /create-embeddings endpoint separately to "
                        "create vector embeddings."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string", "description": "URL to a web page or PDF file"},
                                        "depth": {"type": "integer", "description": "Crawl depth for web pages (ignored for PDFs)"},
                                        "breadth": {"type": "integer", "description": "Max pages to crawl for web pages (ignored for PDFs)"},
                                        "database": {"type": "string", "description": "Database name for storing raw crawled data"},
                                        "collection": {"type": "string", "description": "Collection name for storing raw crawled data"},
                                        "mode": {"type": "string", "enum": ["append", "overwrite"], "description": "How to handle existing data: 'append' adds to existing data (default), 'overwrite' clears existing data first"},
                                    },
                                    "required": ["url", "database", "collection"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Ingestion result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "document_id": {"type": "string"},
                                            "document_type": {"type": "string", "enum": ["web", "pdf"]},
                                            "pages": {"type": "integer"},
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "mode": {"type": "string"},
                                            "message": {"type": "string"},
                                            "overwritten": {"type": "boolean"},
                                            "previous_document_count": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid mode",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/documents": {
                "get": {
                    "summary": "List uploaded documents for a collection",
                    "description": "Retrieve documents from a MongoDB collection. Optionally filter results using a MongoDB query.",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "example": "{\"status\": \"active\"}"},
                            "description": "MongoDB query as JSON object string to filter documents",
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "default": 100, "minimum": 1, "maximum": 10000},
                            "description": "Maximum number of documents to return (default 100, max 10000)",
                        },
                        {
                            "name": "offset",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "default": 0, "minimum": 0},
                            "description": "Number of documents to skip for pagination",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Documents list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {"type": "object"},
                                                "description": "List of documents matching the query",
                                            },
                                            "limit": {"type": "integer", "description": "Page size used"},
                                            "offset": {"type": "integer", "description": "Offset used"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid query JSON",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                },
                "delete": {
                    "summary": "Delete documents from a collection",
                    "description": "Delete documents from a MongoDB collection. Optionally filter deletions using a MongoDB query.",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string", "example": "{\"status\": \"inactive\"}"},
                            "description": "MongoDB query as JSON object string to filter documents",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Deletion result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "deleted_count": {
                                                "type": "integer",
                                                "description": "Number of documents deleted",
                                            },
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters or invalid query JSON",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                },
            },
            "/databases": {
                "get": {
                    "summary": "List all databases with uploaded documents",
                    "responses": {
                        "200": {
                            "description": "Databases list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "databases": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/collections": {
                "get": {
                    "summary": "List collections with uploaded documents",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Collections list"}},
                }
            },
            "/all-databases": {
                "get": {
                    "summary": "List all databases in MongoDB",
                    "description": "Returns all databases in the MongoDB instance, not just those with uploaded documents",
                    "responses": {
                        "200": {
                            "description": "All databases list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "databases": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/all-collections": {
                "get": {
                    "summary": "List all collections in a MongoDB database",
                    "description": "Returns all collections in the specified database, not just those tracked in documents",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "All collections list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collections": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/count-documents": {
                "get": {
                    "summary": "Count documents in a MongoDB collection",
                    "description": "Returns the number of documents in the specified database and collection",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the database",
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The name of the collection",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Document count",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "count": {
                                                "type": "integer",
                                                "description": "The number of documents in the collection",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/write_to_collection": {
                "post": {
                    "summary": "Write a JSON document to a MongoDB collection",
                    "description": "Insert a new document (append mode) or replace an existing document by _id (replace mode). When using replace mode, the update_id parameter is required to specify which document to update.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database_name": {
                                            "type": "string",
                                            "description": "The name of the database",
                                        },
                                        "collection_name": {
                                            "type": "string",
                                            "description": "The name of the collection",
                                        },
                                        "mode": {
                                            "type": "string",
                                            "enum": ["append", "replace"],
                                            "default": "append",
                                            "description": "Write mode: 'append' inserts a new document (default), 'replace' updates an existing document by _id",
                                        },
                                        "document": {
                                            "type": "object",
                                            "description": "The JSON document to write to the collection",
                                        },
                                        "update_id": {
                                            "type": "string",
                                            "description": "The _id of the document to update (required when mode is 'replace')",
                                        },
                                    },
                                    "required": ["database_name", "collection_name", "document"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Document written successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database_name": {"type": "string"},
                                            "collection_name": {"type": "string"},
                                            "mode": {"type": "string"},
                                            "inserted_id": {
                                                "type": "string",
                                                "description": "The _id of the inserted document (append mode only)",
                                            },
                                            "update_id": {
                                                "type": "string",
                                                "description": "The _id of the updated document (replace mode only)",
                                            },
                                            "matched_count": {
                                                "type": "integer",
                                                "description": "Number of documents matched (replace mode only)",
                                            },
                                            "modified_count": {
                                                "type": "integer",
                                                "description": "Number of documents modified (replace mode only)",
                                            },
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing required parameters, invalid mode, or invalid update_id",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "404": {
                            "description": "Document not found (replace mode only)",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/category-mapping": {
                "get": {
                    "summary": "List category mappings",
                    "description": "Retrieve category mappings from privacy-compliance.category_mapping. Optionally filter by statute_category, sub_topic, or a full MongoDB query.",
                    "parameters": [
                        {
                            "name": "statute_category",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Filter by statute_category (e.g. consumer_rights, controller_duties)",
                        },
                        {
                            "name": "sub_topic",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Filter by sub_topic (e.g. data_minimization, privacy_notice)",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "MongoDB query as JSON string to filter documents",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Category mappings list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "category_mappings": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "_id": {"type": "string"},
                                                        "statute_category": {"type": "string"},
                                                        "policy_categories": {"type": "array", "items": {"type": "string"}},
                                                        "sub_topic": {"type": "string"},
                                                        "description": {"type": "string"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid query JSON"},
                    },
                },
                "post": {
                    "summary": "Create a category mapping",
                    "description": "Create a new category mapping in privacy-compliance.category_mapping. Maps statute categories and sub_topics to policy categories.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "statute_category": {"type": "string", "description": "Statute category (e.g. consumer_rights, controller_duties)"},
                                        "policy_categories": {"type": "array", "items": {"type": "string"}, "description": "Policy categories this mapping applies to"},
                                        "sub_topic": {"type": "string", "description": "Optional sub_topic for finer-grained mapping"},
                                        "description": {"type": "string", "description": "Optional description"},
                                    },
                                    "required": ["statute_category", "policy_categories"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Category mapping created",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "collection": {"type": "string"},
                                            "inserted_id": {"type": "string"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters or invalid policy_categories"},
                    },
                },
                "delete": {
                    "summary": "Delete category mappings",
                    "description": "Delete category mappings. Use _id to delete a single document, or query to delete multiple documents.",
                    "parameters": [
                        {
                            "name": "_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "MongoDB ObjectId of the document to delete",
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "MongoDB query as JSON string to delete multiple documents",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Deletion result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "collection": {"type": "string"},
                                            "deleted_count": {"type": "integer"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing _id or query, or invalid parameters"},
                    },
                },
            },
            "/create-embeddings": {
                "post": {
                    "summary": "Index collection rows by embedding text column",
                    "description": "Index all rows in a source collection by embedding the text from the specified column and writing the results to an index collection. The embedding model is determined by the index_database_name.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "source_database_name": {
                                            "type": "string",
                                            "description": "The name of the database containing the collection to be indexed",
                                        },
                                        "source_collection_name": {
                                            "type": "string",
                                            "description": "The name of the collection containing the data to be indexed",
                                        },
                                        "index_database_name": {
                                            "type": "string",
                                            "description": "The database of the index collection",
                                        },
                                        "index_collection_name": {
                                            "type": "string",
                                            "description": "The indexed data will be written to this collection",
                                        },
                                        "text_column": {
                                            "type": "string",
                                            "default": "chunk_text",
                                            "description": "Column to read text from and write embedded text to (default: chunk_text). For subchunks, use subchunk_text so chunk_text from source is preserved.",
                                        },
                                        "source_query": {
                                            "type": "string",
                                            "description": (
                                                "Optional Mongo query as JSON text to filter source rows"
                                            ),
                                        },
                                    },
                                    "required": [
                                        "source_database_name",
                                        "source_collection_name",
                                        "index_database_name",
                                        "index_collection_name",
                                    ],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Indexing result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "source_database_name": {"type": "string"},
                                            "source_collection_name": {"type": "string"},
                                            "index_database_name": {"type": "string"},
                                            "index_collection_name": {"type": "string"},
                                            "text_column": {"type": "string"},
                                            "chunks_indexed": {"type": "integer"},
                                            "embedding_model": {"type": "string"},
                                            "skipped_rows": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters, invalid parameters, or embedding model not configured"},
                        "404": {"description": "Document not found"},
                    },
                }
            },
            "/search": {
                "get": {
                    "summary": "Search vector-indexed collection",
                    "parameters": [
                        {
                            "name": "document_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Search results"}},
                }
            },
            "/vector-search": {
                "get": {
                    "summary": "Vector search over MongoDB Atlas index",
                    "description": "Runs $vectorSearch. Provide query (embedded) or query_vector (precomputed). Model from web-gather.embedding_model when using query. Requires embedding_model configured (POST /embedding-models).",
                    "parameters": [
                        {"name": "database", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "collection", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "index", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "query", "in": "query", "schema": {"type": "string"}, "description": "Search text (embedded). Omit if query_vector provided."},
                        {"name": "query_vector", "in": "query", "schema": {"type": "array", "items": {"type": "number"}}, "description": "Precomputed vector. Omit if query provided."},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
                        {"name": "path", "in": "query", "schema": {"type": "string"}, "description": "Vector field path (default from embedding_model)"},
                        {"name": "filter", "in": "query", "schema": {"type": "string"}, "description": "MongoDB filter as JSON (e.g. {\"jurisdiction\": \"CA\"})"},
                    ],
                    "responses": {"200": {"description": "Vector search results with score"}},
                },
                "post": {
                    "summary": "Vector search over MongoDB Atlas index",
                    "description": "Same as GET. Use POST for long queries or query_vector. Body: database, collection, index; query or query_vector; optional limit, path, filter.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "collection": {"type": "string"},
                                        "index": {"type": "string"},
                                        "query": {"type": "string"},
                                        "query_vector": {"type": "array", "items": {"type": "number"}, "description": "Precomputed vector (skips embedding)"},
                                        "limit": {"type": "integer", "default": 10},
                                        "path": {"type": "string"},
                                        "filter": {"type": "object", "description": "MongoDB filter for $vectorSearch"},
                                    },
                                    "required": ["database", "collection", "index"],
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Vector search results with score"}},
                },
            },
            "/create-paragraph-sections": {
                "post": {
                    "summary": "Split column into paragraph subsections in new collection",
                    "description": "For each record in source_collection (optionally filtered by source_query), splits the column by paragraph boundaries (double newlines) and creates one record per subchunk in destination_collection. Each destination record has all source columns except the split column, plus subsection_column and a unique subchunk_id (guid). Example: database=privacy-compliance, source_collection=statute_chunks, destination_collection=statute_subchunks, column=chunk_text, subsection_column=subchunk_text, source_query={\"jurisdiction\": \"California\"}.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "source_collection": {"type": "string", "description": "Source collection to read from"},
                                        "destination_collection": {"type": "string", "description": "New collection to write subchunks to"},
                                        "column": {"type": "string", "description": "Source field to split (e.g. chunk_text)"},
                                        "subsection_column": {"type": "string", "description": "Field name for subchunk text in destination (e.g. subchunk_text)"},
                                        "source_query": {"type": "object", "description": "Optional MongoDB query to filter source records (e.g. {\"document_id\": \"x\"}). When omitted, all records are processed."},
                                    },
                                    "required": ["database", "source_collection", "destination_collection", "column", "subsection_column"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Records inserted and counts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "source_collection": {"type": "string"},
                                            "destination_collection": {"type": "string"},
                                            "column": {"type": "string"},
                                            "subsection_column": {"type": "string"},
                                            "records_inserted": {"type": "integer"},
                                            "source_rows_processed": {"type": "integer"},
                                            "source_rows_skipped": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters"},
                        "500": {"description": "Failed to read source collection"},
                    },
                }
            },
            "/create-statute-subsections": {
                "post": {
                    "summary": "Split statute section column into subsections using LLM",
                    "description": "For each record in source_collection (optionally filtered by source_query), reads the column value and uses an LLM to identify statute sections by alphabetic markers (a), (b), (c), (d) only. Numeric markers (1), (2), (8) are nested and kept together. The LLM returns line ranges (start_line, end_line) plus metadata (header_text, category, category_reasoning); the backend extracts subsection text from the original document by line numbers and writes one record per section to destination_collection. Optional source_query, parse_prompt.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "source_collection": {"type": "string", "description": "Source collection to read from"},
                                        "destination_collection": {"type": "string", "description": "Collection to write subsections to"},
                                        "column": {"type": "string", "description": "Source field containing statute text (e.g. chunk_text)"},
                                        "subsection_column": {"type": "string", "description": "Field name for subsection text in destination (e.g. subchunk_text)"},
                                        "source_query": {"type": "object", "description": "Optional MongoDB query to filter source records."},
                                        "parse_prompt": {"type": "string", "description": "Optional. Additional parsing instructions. When blank, uses default prompt."},
                                    },
                                    "required": ["database", "source_collection", "destination_collection", "column", "subsection_column"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Records inserted and counts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "source_collection": {"type": "string"},
                                            "destination_collection": {"type": "string"},
                                            "column": {"type": "string"},
                                            "subsection_column": {"type": "string"},
                                            "records_inserted": {"type": "integer"},
                                            "source_rows_processed": {"type": "integer"},
                                            "source_rows_skipped": {"type": "integer"},
                                            "llm_errors": {"type": "integer"},
                                        },
                                    }
                                }
                            }
                        },
                        "400": {"description": "Missing required parameters or invalid source_query JSON"},
                        "500": {"description": "Failed to read source collection"},
                    },
                }
            },
            "/create-statute-subtopics": {
                "post": {
                    "summary": "Identify compliance sub_topics from statute sections using LLM",
                    "description": "For each record in source_collection (optionally filtered by source_query), reads the column value and uses an LLM to identify distinct regulatory sub_topics (testable compliance requirements). Creates one record per sub_topic in destination_collection. Each destination record includes sub_topic, requirement_summary, policy_categories, requires_consent, consumer_facing, and the statute text in subsection_column. Optional parse_prompt: when provided, appended as additional instructions; when blank, uses the default prompt.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "source_collection": {"type": "string", "description": "Source collection to read from"},
                                        "destination_collection": {"type": "string", "description": "Collection to write sub_topics to"},
                                        "column": {"type": "string", "description": "Source field containing statute text (e.g. chunk_text or sub_chunk_text)"},
                                        "subsection_column": {"type": "string", "description": "Field name for statute text in destination (provides context per record)"},
                                        "source_query": {"type": "object", "description": "Optional MongoDB query to filter source records."},
                                        "parse_prompt": {"type": "string", "description": "Optional. Additional parsing instructions. When blank, uses default prompt."},
                                    },
                                    "required": ["database", "source_collection", "destination_collection", "column", "subsection_column"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Records inserted and counts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "source_collection": {"type": "string"},
                                            "destination_collection": {"type": "string"},
                                            "column": {"type": "string"},
                                            "subsection_column": {"type": "string"},
                                            "records_inserted": {"type": "integer"},
                                            "source_rows_processed": {"type": "integer"},
                                            "source_rows_skipped": {"type": "integer"},
                                            "llm_errors": {"type": "integer"},
                                        },
                                    }
                                }
                            }
                        },
                        "400": {"description": "Missing required parameters or invalid source_query JSON"},
                        "500": {"description": "Failed to read source collection"},
                    },
                }
            },
            "/parse-policy-subsections": {
                "post": {
                    "summary": "Parse policy section column into subsections (return in response)",
                    "description": "For each record in collection (optionally filtered by source_query), reads the column value and uses an LLM to identify policy sections (logical chunks). Returns subsections in the response instead of writing to a collection. Optional parse_prompt: when provided, appended as additional instructions.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "collection": {"type": "string", "description": "Source collection to read from"},
                                        "column": {"type": "string", "description": "Source field containing policy text (e.g. chunk_text)"},
                                        "source_query": {"type": "object", "description": "Optional MongoDB query to filter source records."},
                                        "parse_prompt": {"type": "string", "description": "Optional. Additional parsing instructions. When blank, uses default prompt."},
                                    },
                                    "required": ["database", "collection", "column"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Parsed subsections and counts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "collection": {"type": "string"},
                                            "column": {"type": "string"},
                                            "subsections": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "source_id": {"type": "string"},
                                                        "subsection_identifier": {"type": "string"},
                                                        "subsection_text": {"type": "string"},
                                                        "heading": {"type": "string"},
                                                        "category": {"type": "string"},
                                                        "start_line": {"type": "integer"},
                                                        "end_line": {"type": "integer"},
                                                    },
                                                },
                                            },
                                            "source_rows_processed": {"type": "integer"},
                                            "source_rows_skipped": {"type": "integer"},
                                            "llm_errors": {"type": "integer"},
                                        },
                                    }
                                }
                            }
                        },
                        "400": {"description": "Missing required parameters or invalid source_query JSON"},
                        "500": {"description": "Failed to read source collection"},
                    },
                }
            },
            "/create-chunks": {
                "post": {
                    "summary": "Chunk source column with overlap into destination collection",
                    "description": "For each record in source_collection, reads source_column, splits into overlapping chunks (chunk_size, overlap) using character-based chunking, and writes one record per chunk to destination_collection. Each record has all source columns except the source column, plus chunk_column (chunk text), source_id, and chunk_index. Example: database=privacy-compliance, source_collection=documents, destination_collection=chunks, chunk_size=1200, overlap=200, source_column=text, chunk_column=chunk_text.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "source_collection": {"type": "string", "description": "Source collection to read from"},
                                        "destination_collection": {"type": "string", "description": "Collection to write chunks to"},
                                        "chunk_size": {"type": "integer", "default": 1200, "description": "Max characters per chunk"},
                                        "overlap": {"type": "integer", "default": 200, "description": "Overlap between chunks"},
                                        "source_column": {"type": "string", "description": "Field to read text from (e.g. text)"},
                                        "chunk_column": {"type": "string", "description": "Field to write chunk text to (e.g. chunk_text)"},
                                    },
                                    "required": ["database", "source_collection", "destination_collection", "source_column", "chunk_column"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Records inserted and counts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "database": {"type": "string"},
                                            "source_collection": {"type": "string"},
                                            "destination_collection": {"type": "string"},
                                            "chunk_size": {"type": "integer"},
                                            "overlap": {"type": "integer"},
                                            "source_column": {"type": "string"},
                                            "chunk_column": {"type": "string"},
                                            "records_inserted": {"type": "integer"},
                                            "source_rows_processed": {"type": "integer"},
                                            "source_rows_skipped": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters or invalid chunk_size/overlap"},
                        "500": {"description": "Failed to read source collection"},
                    },
                }
            },
            "/embedding-models": {
                "get": {
                    "summary": "List embedding models",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Embedding models"}},
                },
                "post": {
                    "summary": "Add embedding model",
                    "description": "Upsert an embedding_model document by database_name. Requires database_name, model_name, and fields (vector index definitions). Any additional top-level keys are stored as-is.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "database_name": {"type": "string"},
                                        "model_name": {"type": "string"},
                                        "fields": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "string", "enum": ["vector"]},
                                                    "path": {"type": "string"},
                                                    "numDimensions": {"type": "number"},
                                                    "similarity": {"type": "string", "enum": ["cosine"], "description": "Must be cosine (euclidean/dotProduct not supported)."},
                                                },
                                                "required": ["type", "path", "numDimensions", "similarity"],
                                            },
                                        },
                                    },
                                    "required": ["database_name", "model_name", "fields"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Embedding model saved"}},
                },
            },
            "/create-vector-index": {
                "post": {
                    "summary": "Create vector search index from embedding_model",
                    "description": "Create an Atlas vector search index on a collection using the embedding_model document from web-gather (database_name). Drops the index first if it exists. Body: database_name, collection_name, optional index_name (default vector_index). Requires Atlas.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database_name": {"type": "string"},
                                        "collection_name": {"type": "string"},
                                        "index_name": {"type": "string"},
                                        "filter_fields": {
                                            "type": "array",
                                            "description": "Field paths to index for pre-filtering (e.g. document_id, jurisdiction). If omitted, uses embedding_model.filter_fields when present.",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}
                                                ]
                                            },
                                        },
                                    },
                                    "required": ["database_name", "collection_name"],
                                },
                                "example": {
                                    "database_name": "compliance",
                                    "collection_name": "statute_sub_embeddings",
                                    "index_name": "vector_index",
                                    "filter_fields": ["jurisdiction", "document_id"]
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Index creation started"},
                        "400": {"description": "Missing embedding_model or invalid body"},
                        "500": {"description": "createSearchIndexes failed"},
                    },
                },
            },
            "/create-sub-vector-index": {
                "post": {
                    "summary": "Start background job to create sub-vector indexes",
                    "description": "Creates embeddings and vector index for statute or policy chunks. Statute: subsections -> subtopics -> statute_sub_embeddings. Policy: policy_chunks -> policy_legal_embeddings (no subsections). Runs as a background LangChain workflow. Returns job_id immediately. Use GET /index-jobs/{job_id} to poll status. For policy, provide source_query with document_id to index a specific policy.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_type": {
                                            "type": "string",
                                            "enum": ["policy", "statute"],
                                            "description": "Type of document to process",
                                        },
                                        "source_query": {
                                            "oneOf": [
                                                {"type": "object", "description": "MongoDB query object"},
                                                {"type": "string", "description": "JSON string of MongoDB query"},
                                            ],
                                            "description": "MongoDB query to filter source records. For policy, use {\"document_id\": \"<policy-uuid>\"} to index a specific policy.",
                                        },
                                    },
                                    "required": ["document_type"],
                                },
                                "example": {
                                    "document_type": "policy",
                                    "source_query": {"document_id": "b13529b1-c5ad-4e5d-8a44-d4557ebde360"},
                                },
                            }
                        },
                    },
                    "responses": {
                        "202": {
                            "description": "Job started",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "status": {"type": "string", "example": "pending"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid document_type or source_query"},
                        "500": {"description": "Index job service not initialized"},
                    },
                },
            },
            "/index-jobs/{job_id}": {
                "get": {
                    "summary": "Get index job status",
                    "description": "Returns job status, result (when completed), or error (when failed).",
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Job ID returned from POST /create-sub-vector-index",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Job status and result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "job_type": {"type": "string"},
                                            "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
                                            "request": {"type": "object"},
                                            "result": {"type": "object"},
                                            "error": {"type": "string"},
                                            "created_at": {"type": "string"},
                                            "started_at": {"type": "string"},
                                            "completed_at": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "404": {"description": "Job not found"},
                    },
                },
            },
            "/parse-llm": {
                "post": {
                    "summary": "Parse document text from a collection using LLM",
                    "description": "Reads records from the specified database/collection and uses this as input for the LLM along with the parse_prompt. If document_id is provided, only that specific document is parsed; otherwise, all documents in the collection are concatenated. Returns structured JSON with parsed sections.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database": {
                                            "type": "string",
                                            "description": "The name of the database containing the documents to parse",
                                        },
                                        "collection": {
                                            "type": "string",
                                            "description": "The name of the collection containing the documents to parse",
                                        },
                                        "parse_prompt": {
                                            "type": "string",
                                            "description": "Instructions for how to parse the combined document text",
                                        },
                                        "document_id": {
                                            "type": "string",
                                            "description": "Optional MongoDB ObjectId of a specific document to parse. If not provided, all documents in the collection are concatenated for parsing.",
                                        },
                                    },
                                    "required": ["database", "collection", "parse_prompt"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Parsed document sections",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "parsed_doc": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "section": {"type": "string"},
                                                        "code_name": {"type": "string"},
                                                        "jurisdiction": {"type": "string"},
                                                        "parsed_header_text": {"type": "string"},
                                                        "parsed_text": {"type": "string"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters or no documents found"},
                        "500": {"description": "LLM parsing failed"},
                    },
                }
            },
            "/gap-check": {
                "post": {
                    "summary": "Single statute-to-policy gap check",
                    "description": "Fetches both the statute requirement and policy document from MongoDB, invokes the LLM to determine if the policy addresses the statute requirement, and returns a structured gap_check result (addressed, policy_quote, missing, conflict, conflict_description).",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GapCheckRequest"},
                                "example": {
                                    "policy_database_name": "privacy-compliance",
                                    "policy_collection_name": "policies",
                                    "policy_document_id": "142bcbe4-f34b-4f60-8be3-79ed269375ab",
                                    "statute_database_name": "privacy-compliance",
                                    "statute_collection_name": "statutes",
                                    "statute_document_id": "8df572eb-4898-4677-98bf-25e96a7701fa",
                                    "max_policy_chars": 8000,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Gap check result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GapCheckResponse"},
                                }
                            },
                        },
                        "400": {
                            "description": "Missing required parameters",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                        "404": {
                            "description": "Statute or policy document not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                        "500": {
                            "description": "LLM parsing failed or could not parse LLM response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                                }
                            },
                        },
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/statute-policy-compliance": {
                "post": {
                    "summary": "Statute-first compliance check",
                    "description": "Iterates statutory requirements via category mappings, finds matching policy chunks in policy_legal_embeddings using vector search, and evaluates each requirement for compliance gaps. Returns per-requirement gap items with status, quotes, and a summary.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/StatutePolicyComplianceRequest"},
                                "example": {
                                    "policy_id": "doc-123",
                                    "jurisdiction": "CA",
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Statute-policy compliance analysis response with gap items",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/StatutePolicyComplianceResponse"}
                                }
                            },
                        },
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "401": {"description": "Missing API key", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "403": {"description": "Unauthorized", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "500": {"description": "Internal server error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/applicability": {
                "post": {
                    "summary": "Infer applicable jurisdictions",
                    "description": "Analyze policy text to infer which US state codes (e.g. CA, VA) or US federal the policy is likely intended for.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ApplicabilityRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Applicable jurisdictions and confidence",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ApplicabilityResponse"},
                                }
                            },
                        },
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/gap-analysis": {
                "post": {
                    "summary": "Full gap analysis (subchunk)",
                    "description": "Retrieve relevant statute subchunks for the policy's jurisdictions, run gap checks against each, and return a full gap analysis with gaps, summary, and optional persistence. Uses statute_sub_embeddings and policy_sub_embeddings.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GapAnalysisRequest"},
                                "example": {
                                    "policy_document_id": "doc-123",
                                    "applicable_jurisdictions": ["CA", "VA"],
                                    "num_rows": 10,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Gap analysis with gaps and summary (synchronous when run_async=false)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GapAnalysisResponse"},
                                }
                            },
                        },
                        "202": {
                            "description": "Job started (when run_async=true). Returns job_id; poll GET /jobs/{job_id} for result.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "status": {"type": "string", "example": "pending"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/jobs/{{job_id}}": {
                "get": {
                    "summary": "Get compliance job status",
                    "description": "Return status and result of a background compliance job (gap analysis, health score, etc.). Poll this endpoint after starting a job with run_async=true.",
                    "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {
                        "200": {
                            "description": "Job status and result (when completed)",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ComplianceJobResponse"}}},
                        },
                        "404": {"description": "Job not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_V2_API_PREFIX}/gap-analysis": {
                "post": {
                    "summary": "Gap analysis (chunk-level, v2)",
                    "description": "Chunk-level gap analysis: compares statute_embeddings to policy_embeddings. Uses a YAML-configurable prompt (prompts/gap_analysis_chunk.yaml). Different from v1 which uses statute_sub_embeddings and policy_sub_embeddings.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GapAnalysisRequest"},
                                "example": {
                                    "policy_document_id": "doc-123",
                                    "applicable_jurisdictions": ["CA", "VA"],
                                    "num_rows": 10,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Gap analysis with gaps and summary",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GapAnalysisResponse"},
                                }
                            },
                        },
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_V3_API_PREFIX}/gap-analysis": {
                "post": {
                    "summary": "Gap analysis (v3)",
                    "description": "Gap analysis v3 per GapAnalysisProcessDesign.md (v1 design): Uses statute_sub_embeddings and policy_sub_embeddings. Subchunks include parent context; no full policy text in prompt. Statute→policy vector search with top-k matches and score threshold, LLM analysis per pair, citation binding validation against full policy text.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GapAnalysisRequest"},
                                "examples": {
                                    "single": {
                                        "summary": "Single policy document",
                                        "value": {
                                            "policy_document_id": "doc-123",
                                            "applicable_jurisdictions": ["CA", "VA"],
                                            "num_rows": 10,
                                            "save_results": True,
                                        },
                                    },
                                    "multiple": {
                                        "summary": "Multiple policy documents (v3)",
                                        "value": {
                                            "policy_document_ids": ["doc-123", "doc-456"],
                                            "applicable_jurisdictions": ["CA", "VA"],
                                            "num_rows": 10,
                                            "save_results": True,
                                        },
                                    },
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Gap analysis with gaps, summary, retrieval_metadata (statute_items_considered, statute_pairs_matched)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GapAnalysisResponse"},
                                }
                            },
                        },
                        "202": {
                            "description": "Job started (when run_async=true). Returns job_id; poll GET /api/compliance/jobs/{job_id} for result.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "status": {"type": "string", "example": "pending"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request (e.g. policy not indexed)", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_V4_API_PREFIX}/gap-analysis": {
                "post": {
                    "summary": "Gap analysis (v4)",
                    "description": "Gap analysis v4: Category-mapping-driven approach. Uses statute_sub_topic_embeddings and policy_legal_embeddings with category_mapping to drive comparisons. Injects definitions and applicability as reference context when analyzing consumer_rights and controller_duties requirements.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GapAnalysisRequest"},
                                "examples": {
                                    "single": {
                                        "summary": "Single policy document",
                                        "value": {
                                            "policy_document_id": "doc-123",
                                            "applicable_jurisdictions": ["CA", "VA"],
                                            "num_rows": 10,
                                            "save_results": True,
                                        },
                                    },
                                    "multiple": {
                                        "summary": "Multiple policy documents (v4)",
                                        "value": {
                                            "policy_document_ids": ["doc-123", "doc-456"],
                                            "applicable_jurisdictions": ["CA", "VA"],
                                            "num_rows": 10,
                                            "save_results": True,
                                        },
                                    },
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Gap analysis with gaps, summary, retrieval_metadata (statute_items_considered, statute_pairs_matched)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GapAnalysisResponse"},
                                }
                            },
                        },
                        "202": {
                            "description": "Job started (when run_async=true). Returns job_id; poll GET /api/compliance/jobs/{job_id} for result.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "status": {"type": "string", "example": "pending"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request (e.g. policy not indexed)", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/multi-jurisdictional": {
                "post": {
                    "summary": "Strictest common denominator",
                    "description": "Compare how each jurisdiction formulates requirements and identify the strictest formulation.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MultiJurisdictionalRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Strictest common denominator analysis",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MultiJurisdictionalResponse"},
                                }
                            },
                        },
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/health-score": {
                "post": {
                    "summary": "Privacy health score",
                    "description": "Compute a 0-100 privacy health score from v4 gap analysis (statute_sub_topic_embeddings, policy_legal_embeddings, consumer_rights/controller_duties only). Uses standard weights by default: consumer rights (1.5), controller duties (1.2), processor duties (1.0). Override via weights in request or requirement_weights in config.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthScoreRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Health score and breakdown (synchronous when run_async=false)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthScoreResponse"},
                                }
                            },
                        },
                        "202": {
                            "description": "Job started (when run_async=true). Returns job_id; poll GET /api/compliance/jobs/{job_id} for result.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "status": {"type": "string", "example": "pending"},
                                            "message": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/drift-check": {
                "post": {
                    "summary": "Regulatory drift check",
                    "description": "Check for new statute chunks since last run and generate drift alerts for affected policies.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DriftCheckRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Drift alerts and counts",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DriftCheckResponse"},
                                }
                            },
                        },
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/report": {
                "post": {
                    "summary": "Generate compliance report",
                    "description": "Generate a Markdown or PDF compliance report for a policy from latest stored result or an ad-hoc run (run_now does not persist).",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ReportRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Report content (markdown)",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReportResponseMarkdown"},
                                }
                            },
                        },
                        "400": {"description": "Invalid request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "No stored result when source=latest_stored", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "501": {"description": "PDF not implemented", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "Upstream (gap/health) failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/runs": {
                "get": {
                    "summary": "List compliance runs",
                    "description": "List versioned compliance runs (audit trail) with optional filters: policy_document_id, since, until, limit, offset, types (gap, health_score, multi_jurisdictional, applicability).",
                    "parameters": [
                        {"name": "policy_document_id", "in": "query", "schema": {"type": "string"}, "description": "Filter by policy."},
                        {"name": "since", "in": "query", "schema": {"type": "string", "format": "date-time"}, "description": "Runs with run_at >= since (ISO8601)."},
                        {"name": "until", "in": "query", "schema": {"type": "string", "format": "date-time"}, "description": "Runs with run_at <= until (ISO8601)."},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50, "maximum": 200}, "description": "Max items."},
                        {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}, "description": "Pagination offset."},
                        {"name": "types", "in": "query", "schema": {"type": "string"}, "description": "Comma-separated: gap, health_score, multi_jurisdictional, applicability."},
                    ],
                    "responses": {
                        "200": {
                            "description": "List of run summaries",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/RunsListResponse"}}},
                        },
                        "400": {"description": "Invalid since/until or limit/offset", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "Upstream failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/runs/{{run_id}}": {
                "get": {
                    "summary": "Get run detail",
                    "description": "Return full payload for a single compliance run.",
                    "parameters": [{"name": "run_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {
                        "200": {
                            "description": "Full run payload (gaps, summary, privacy_health_score, etc.)",
                            "content": {"application/json": {"schema": {"type": "object"}}},
                        },
                        "404": {"description": "Run not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/citations": {
                "post": {
                    "summary": "Statute–policy citation extraction",
                    "description": "Extract statute–policy citations: for each relevant policy section, return policy excerpt, statute reference, and alignment.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CitationsRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Citations and summary",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CitationsResponse"}}},
                        },
                        "400": {"description": "Missing policy_document_id", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "Upstream or LLM failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/risk-assessment": {
                "post": {
                    "summary": "Risk assessment (DPIA/PIA-style)",
                    "description": "Generate a pre-populated risk assessment from policy and applicable statute chunks via LLM.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RiskAssessmentRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Assessment and optional report",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/RiskAssessmentResponse"}}},
                        },
                        "400": {"description": "Missing or invalid request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "404": {"description": "Policy or template not found", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "Upstream or LLM failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/risk-assessment/templates": {
                "get": {
                    "summary": "List risk-assessment templates",
                    "description": "List available risk-assessment template IDs and labels.",
                    "responses": {
                        "200": {
                            "description": "List of templates",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TemplatesResponse"}}},
                        },
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/alerts": {
                "get": {
                    "summary": "List drift alerts",
                    "description": "List regulatory drift alerts from compliance_alerts with filters.",
                    "parameters": [
                        {"name": "policy_document_id", "in": "query", "schema": {"type": "string"}},
                        {"name": "company_name", "in": "query", "schema": {"type": "string"}, "description": "Substring match."},
                        {"name": "jurisdiction", "in": "query", "schema": {"type": "string"}, "description": "Filter by affected jurisdiction."},
                        {"name": "since", "in": "query", "schema": {"type": "string", "format": "date-time"}, "description": "detected_at >= since."},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50, "maximum": 200}},
                        {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
                    ],
                    "responses": {
                        "200": {
                            "description": "List of alerts",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AlertsListResponse"}}},
                        },
                        "400": {"description": "Invalid parameters", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "Upstream failed", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/suggest-policy": {
                "post": {
                    "summary": "Suggest compliant policy text",
                    "description": "Analyze a gap analysis finding and rewrite or add text to a policy to make it compliant with the given statute. Returns the full revised policy text and a description of the modifications.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SuggestPolicyRequest"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Suggested compliant policy text with modification details",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SuggestPolicyResponse"}}},
                        },
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "429": {"description": "Rate limit exceeded", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "502": {"description": "LLM failed to generate suggestion", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            f"{COMPLIANCE_API_PREFIX}/consumer-rights-router": {
                "post": {
                    "summary": "Consumer rights request decision trees",
                    "description": "Analyze a privacy policy against US state privacy statutes to generate per-jurisdiction decision trees for handling consumer data-subject requests (deletion, access, opt-out, correction). Returns decision trees for operational routing and flags policy gaps where the stated handling does not meet statutory requirements. Uses privacy-compliance.policy_legal_embeddings for policy lookup.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ConsumerRightsRouterRequest"},
                                "example": {
                                    "policy_document_id": "doc-123",
                                    "applicable_jurisdictions": ["CA", "VA", "CO", "TX"],
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Decision trees and policy gap analysis per request type per jurisdiction",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ConsumerRightsRouterResponse"}}},
                        },
                        "400": {"description": "Policy not found or empty", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "429": {"description": "Rate limit exceeded", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            "/statute-policy-compliance": {
                "post": {
                    "summary": "Statute-first compliance check (root path)",
                    "description": "Same as /api/compliance/statute-policy-compliance. Iterates statutory requirements via category mappings, finds matching policy chunks, and evaluates each requirement for compliance gaps.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/StatutePolicyComplianceRequest"},
                                "example": {
                                    "policy_id": "doc-123",
                                    "jurisdiction": "CA",
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Statute-policy compliance analysis response", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/StatutePolicyComplianceResponse"}}}},
                        "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "401": {"description": "Missing API key", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "403": {"description": "Unauthorized", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "422": {"description": "Validation error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                        "500": {"description": "Internal server error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                    },
                }
            },
            "/crawl": {
                "post": {
                    "summary": "Crawl a URL with specified depth and breadth",
                    "description": "Crawl a URL and return combined text from all pages visited. Tries Playwright first, then Puppeteer, then Selenium, then Firecrawl on failure. Use depth to control how many levels of links to follow, and breadth to limit the maximum number of pages crawled.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {
                                            "type": "string",
                                            "description": "The URL to start crawling from",
                                        },
                                        "depth": {
                                            "type": "integer",
                                            "description": "How deep to follow links from the starting URL (default: 2)",
                                            "default": 2,
                                            "minimum": 1,
                                        },
                                        "breadth": {
                                            "type": "integer",
                                            "description": "Maximum number of pages to crawl (default: 10)",
                                            "default": 10,
                                            "minimum": 1,
                                        },
                                    },
                                    "required": ["url"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Crawl results with combined page text",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "url": {
                                                "type": "string",
                                                "description": "The starting URL that was crawled",
                                            },
                                            "depth": {
                                                "type": "integer",
                                                "description": "The depth parameter used",
                                            },
                                            "breadth": {
                                                "type": "integer",
                                                "description": "The breadth parameter used",
                                            },
                                            "pages_crawled": {
                                                "type": "integer",
                                                "description": "Number of pages successfully crawled",
                                            },
                                            "urls_crawled": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of URLs that were crawled",
                                            },
                                            "combined_text": {
                                                "type": "string",
                                                "description": "Combined text content from all crawled pages",
                                            },
                                            "text_length": {
                                                "type": "integer",
                                                "description": "Length of the combined text in characters",
                                            },
                                            "crawl_method": {
                                                "type": "string",
                                                "description": "Method used: playwright, puppeteer, selenium, or firecrawl",
                                                "enum": ["playwright", "puppeteer", "selenium", "firecrawl"],
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Bad request - missing URL, invalid parameters, or no content returned",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "500": {
                            "description": "Crawl failed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                    },
                }
            },
        },
    }


@routes_bp.get("/openapi.json")
def openapi():
    """Return the OpenAPI specification."""
    return jsonify(build_openapi_spec())


@routes_bp.get("/docs/")
def docs_trailing():
    """Redirect /docs/ to /docs."""
    return redirect("/docs", code=302)


@routes_bp.get("/docs")
def docs():
    """Render the Swagger UI documentation."""
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Web Gather API Docs</title>
        <link
          rel="stylesheet"
          href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
        />
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
          window.onload = () => {
            SwaggerUIBundle({
              url: "/openapi.json",
              dom_id: "#swagger-ui"
            });
          };
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")
