"""Regression tests for OpenAPI path inventory and security contract.

Ensures build_openapi_spec documents core + compliance v1–v4 routes and keeps
the ApiKeyAuth scheme / health unauthenticated override intact. Spec drift here
breaks Swagger clients and hides auth requirements.
"""
from __future__ import annotations

from routes import (
    COMPLIANCE_API_PREFIX,
    COMPLIANCE_V2_API_PREFIX,
    COMPLIANCE_V3_API_PREFIX,
    COMPLIANCE_V4_API_PREFIX,
    build_openapi_spec,
)


# Paths that must exist for core/db/util surface area (non-compliance).
REQUIRED_CORE_PATHS = {
    "/health",
    "/gather",
    "/ingest",
    "/crawl",
    "/documents",
    "/databases",
    "/collections",
    "/create-embeddings",
    "/search",
    "/vector-search",
    "/create-chunks",
    "/create-paragraph-sections",
    "/create-statute-subsections",
    "/create-statute-subtopics",
    "/parse-policy-subsections",
    "/embedding-models",
    "/create-vector-index",
    "/create-sub-vector-index",
    "/index-jobs/{job_id}",
    "/parse-llm",
    "/gap-check",
    "/statute-policy-compliance",
}

# Compliance routes registered in app.py under versioned prefixes.
REQUIRED_COMPLIANCE_PATHS = {
    f"{COMPLIANCE_API_PREFIX}/statute-policy-compliance",
    f"{COMPLIANCE_API_PREFIX}/applicability",
    f"{COMPLIANCE_API_PREFIX}/gap-analysis",
    f"{COMPLIANCE_API_PREFIX}/jobs/{{job_id}}",
    f"{COMPLIANCE_API_PREFIX}/multi-jurisdictional",
    f"{COMPLIANCE_API_PREFIX}/health-score",
    f"{COMPLIANCE_API_PREFIX}/drift-check",
    f"{COMPLIANCE_API_PREFIX}/report",
    f"{COMPLIANCE_API_PREFIX}/runs",
    f"{COMPLIANCE_API_PREFIX}/runs/{{run_id}}",
    f"{COMPLIANCE_API_PREFIX}/citations",
    f"{COMPLIANCE_API_PREFIX}/risk-assessment",
    f"{COMPLIANCE_API_PREFIX}/risk-assessment/templates",
    f"{COMPLIANCE_API_PREFIX}/alerts",
    f"{COMPLIANCE_API_PREFIX}/suggest-policy",
    f"{COMPLIANCE_API_PREFIX}/consumer-rights-router",
    f"{COMPLIANCE_V2_API_PREFIX}/gap-analysis",
    f"{COMPLIANCE_V3_API_PREFIX}/gap-analysis",
    f"{COMPLIANCE_V4_API_PREFIX}/gap-analysis",
    f"{COMPLIANCE_V4_API_PREFIX}/adaptive-feedback",
    f"{COMPLIANCE_V4_API_PREFIX}/adaptive-feedback/log",
}


def test_openapi_security_scheme_is_api_key_header() -> None:
    """Global ApiKeyAuth must remain x-api-key header auth for clients."""
    spec = build_openapi_spec()
    schemes = spec["components"]["securitySchemes"]
    assert "ApiKeyAuth" in schemes
    auth = schemes["ApiKeyAuth"]
    assert auth["type"] == "apiKey"
    assert auth["in"] == "header"
    assert auth["name"] == "x-api-key"
    assert spec["security"] == [{"ApiKeyAuth": []}]


def test_openapi_health_explicitly_disables_auth() -> None:
    """Health checks must stay publicly reachable even when API keys are configured."""
    spec = build_openapi_spec()
    health = spec["paths"]["/health"]["get"]
    assert health["security"] == []


def test_openapi_includes_required_core_and_compliance_paths() -> None:
    """Spec must document core + compliance v1–v4 routes actually mounted by app.py."""
    spec = build_openapi_spec()
    paths = set(spec["paths"])
    missing_core = sorted(REQUIRED_CORE_PATHS - paths)
    missing_compliance = sorted(REQUIRED_COMPLIANCE_PATHS - paths)
    assert missing_core == [], f"Missing core OpenAPI paths: {missing_core}"
    assert missing_compliance == [], f"Missing compliance OpenAPI paths: {missing_compliance}"


def test_openapi_gap_analysis_versions_declare_post() -> None:
    """Each versioned gap-analysis path must expose POST (sync/async entrypoint)."""
    spec = build_openapi_spec()
    for path in (
        f"{COMPLIANCE_API_PREFIX}/gap-analysis",
        f"{COMPLIANCE_V2_API_PREFIX}/gap-analysis",
        f"{COMPLIANCE_V3_API_PREFIX}/gap-analysis",
        f"{COMPLIANCE_V4_API_PREFIX}/gap-analysis",
    ):
        assert "post" in spec["paths"][path], f"{path} missing post operation"


def test_openapi_compliance_job_and_adaptive_feedback_get_ops() -> None:
    """Job polling and adaptive-feedback history/log are GET endpoints in the spec."""
    spec = build_openapi_spec()
    assert "get" in spec["paths"][f"{COMPLIANCE_API_PREFIX}/jobs/{{job_id}}"]
    assert "get" in spec["paths"][f"{COMPLIANCE_V4_API_PREFIX}/adaptive-feedback"]
    assert "get" in spec["paths"][f"{COMPLIANCE_V4_API_PREFIX}/adaptive-feedback/log"]
