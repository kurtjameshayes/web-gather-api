"""Configuration loader for statute-policy compliance service."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class ConfidenceThresholds:
    compliant: float
    non_compliant: float


@dataclass
class ComplianceConfig:
    top_k_statutes: int
    confidence_thresholds: ConfidenceThresholds
    evidence_score_threshold: float
    max_statute_chunk_chars: int
    max_section_chars: int
    statutes_collection: str
    embeddings_collection: str
    use_embeddings_collection: bool
    vector_index_name: str
    vector_field: str
    statute_text_field: str
    statute_title_field: str
    statute_id_field: str
    statute_section_id_field: str
    statute_section_field: str  # Formal citation (e.g. § 1798.100) in statute_chunks/embeddings
    statute_jurisdiction_field: str
    statute_corpus_field: str
    embedding_collection_tag_field: str
    embedding_vector_field: str
    embedding_text_field: str
    embedding_doc_id_field: str
    embedding_chunk_id_field: str
    cache_size: int
    cache_ttl_seconds: int
    llm_max_tokens: int
    llm_concurrency: int
    rate_limit_per_minute: int
    audit_collection: str
    enable_redaction: bool
    enable_audit_logging: bool
    allow_raw_audit: bool
    auth_required: bool
    api_key: str
    allowed_roles: List[str]
    default_role_header: str
    embedding_model_name: str
    llm_model_name: str
    statute_collection_tag: str
    # Database for statute/embedding retrieval (when set; else compliance_database)
    statute_database: str
    # Compliance suite (gap analysis, health score, drift)
    compliance_database: str
    policies_collection: str
    policy_chunks_collection: str
    policy_document_id_field: str
    policy_chunk_index_field: str
    policy_chunk_text_field: str
    policy_chunk_header_field: str
    statute_chunk_header_field: str
    compliance_results_collection: str
    compliance_alerts_collection: str
    compliance_run_log_collection: str
    compliance_jobs_collection: str
    index_jobs_collection: str
    statute_index_version_field: str
    max_statute_quote_chars: int
    default_jurisdictions: List[str]
    conflict_penalty_multiplier: float
    partial_credit_percent: float  # 0.0-1.0; partial status contributes this fraction to health score
    requirement_weights: Dict[str, float]
    canonical_requirement_ids: List[str]
    disclosure_queries: List[str]
    # Chunk-level gap analysis (v2: statute_embeddings vs policy_embeddings)
    statute_embeddings_collection: str
    policy_embeddings_collection: str
    gap_analysis_chunk_prompt_path: str
    # Subchunk gap analysis
    statute_sub_embeddings_collection: str
    policy_sub_embeddings_collection: str
    statute_subchunk_text_field: str
    policy_subchunk_text_field: str
    statute_chunk_text_field: str  # Enclosing chunk in statute subchunks
    # Gap analysis v3 (design: statute→policy vector search, top-k, score threshold)
    gap_analysis_v3_top_k: int
    gap_analysis_v3_num_candidates: int
    gap_analysis_v3_score_threshold: float
    policy_embeddings_vector_index: str  # Vector index name for policy_embeddings
    policy_sub_embeddings_vector_index: str  # Vector index for policy_sub_embeddings (v1)
    gap_analysis_v3_prompt_path: str
    # Gap analysis v4 (category-mapping-driven: statute_sub_topic_embeddings, policy_legal_embeddings)
    statute_sub_topic_embeddings_collection: str
    policy_legal_embeddings_collection: str
    category_mapping_collection: str
    gap_analysis_v4_prompt_path: str
    consumer_rights_router_prompt_path: str
    # Adaptive feedback loop (v4 gap analysis)
    adaptive_feedback_collection: str
    adaptive_feedback_log_collection: str
    adaptive_feedback_enabled: bool
    adaptive_feedback_max_items: int
    adaptive_critic_prompt_path: str


DEFAULT_CONFIG: Dict[str, Any] = {
    "top_k_statutes": 5,
    "confidence_thresholds": {"compliant": 0.75, "non_compliant": 0.75},
    "evidence_score_threshold": 0.2,
    "max_statute_chunk_chars": 1200,
    "max_section_chars": 2000,
    "statutes_collection": "statutes",
    "embeddings_collection": "embeddings",
    "use_embeddings_collection": False,
    "vector_index_name": "vector_index",
    "vector_field": "vector",
    "statute_text_field": "section_text",
    "statute_title_field": "title",
    "statute_id_field": "_id",
    "statute_section_id_field": "section_id",
    "statute_section_field": "section",
    "statute_jurisdiction_field": "jurisdiction",
    "statute_corpus_field": "statute_corpus_id",
    "embedding_collection_tag_field": "collection",
    "embedding_vector_field": "vector",
    "embedding_text_field": "text",
    "embedding_doc_id_field": "doc_id",
    "embedding_chunk_id_field": "chunk_id",
    "cache_size": 512,
    "cache_ttl_seconds": 3600,
    "llm_max_tokens": 1024,
    "llm_concurrency": 4,
    "rate_limit_per_minute": 60,
    "audit_collection": "policy_compliance_audit",
    "enable_redaction": True,
    "enable_audit_logging": True,
    "allow_raw_audit": False,
    "auth_required": False,
    "api_key": "",
    "allowed_roles": ["admin", "compliance"],
    "default_role_header": "x-role",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "llm_model_name": "claude-sonnet-4-6",
    "statute_collection_tag": "statutes",
    "statute_database": "",
    # Compliance suite
    "compliance_database": "privacy-compliance",
    "policies_collection": "policies",
    "policy_chunks_collection": "policy_chunks",
    "policy_document_id_field": "document_id",
    "policy_chunk_index_field": "chunk_index",
    "policy_chunk_text_field": "chunk_text",
    "policy_chunk_header_field": "chunk_header_text",
    "statute_chunk_header_field": "chunk_header_text",
    "compliance_results_collection": "compliance_results",
    "compliance_alerts_collection": "compliance_alerts",
    "compliance_run_log_collection": "compliance_run_log",
    "compliance_jobs_collection": "compliance_jobs",
    "index_jobs_collection": "index_job",
    "statute_index_version_field": "indexed_at",
    "max_statute_quote_chars": 300,
    "default_jurisdictions": ["CA", "VA", "CO", "CT"],
    "conflict_penalty_multiplier": 0.7,
    "partial_credit_percent": 0.5,
    "requirement_weights": {
        # Consumer rights (core CCPA/CPRA-style protections) - weight 1.5
        "right to know": 1.5,
        "right to access": 1.5,
        "right to delete": 1.5,
        "right to correct": 1.5,
        "opt out of sale": 1.5,
        "opt-out of sale": 1.5,
        "opt out of sharing": 1.5,
        "opt-out of sharing": 1.5,
        "sensitive data": 1.5,
        "sensitive personal information": 1.5,
        "non-discrimination": 1.5,
        "non discrimination": 1.5,
        "do not sell": 1.5,
        "do not share": 1.5,
        # Controller duties - weight 1.2
        "privacy notice": 1.2,
        "data minimization": 1.2,
        "purpose limitation": 1.2,
        "data security": 1.2,
        "retention": 1.2,
        "data retention": 1.2,
        # Processor duties - weight 1.0
        "processor must": 1.0,
        "processor shall": 1.0,
        "processor duties": 1.0,
    },
    "canonical_requirement_ids": [
        "right_to_know",
        "right_to_delete",
        "opt_out_of_sale",
        "sensitive_data",
        "non_discrimination",
    ],
    "disclosure_queries": [
        "right to know what personal information is collected",
        "right to delete personal information",
        "opt out of sale of personal data",
        "sensitive data disclosure and consent",
    ],
    # Chunk-level gap analysis (v2)
    "statute_embeddings_collection": "statute_embeddings",
    "policy_embeddings_collection": "policy_embeddings",
    "gap_analysis_chunk_prompt_path": "prompts/gap_analysis_chunk.yaml",
    # Subchunk gap analysis
    "statute_sub_embeddings_collection": "statute_sub_embeddings",
    "policy_sub_embeddings_collection": "policy_sub_embeddings",
    "statute_subchunk_text_field": "sub_chunk_text",
    "policy_subchunk_text_field": "sub_chunk_text",
    "statute_chunk_text_field": "chunk_text",
    # Gap analysis v3
    "gap_analysis_v3_top_k": 5,
    "gap_analysis_v3_num_candidates": 50,
    "gap_analysis_v3_score_threshold": 0.70,
    "policy_embeddings_vector_index": "policy_embeddings_vector_index",
    "policy_sub_embeddings_vector_index": "policy_sub_embeddings_vector_index",
    "gap_analysis_v3_prompt_path": "prompts/gap_analysis_v3.yaml",
    # Gap analysis v4
    "statute_sub_topic_embeddings_collection": "statute_sub_topic_embeddings",
    "policy_legal_embeddings_collection": "policy_legal_embeddings",
    "category_mapping_collection": "category_mapping",
    "gap_analysis_v4_prompt_path": "prompts/gap_analysis_v4.yaml",
    "consumer_rights_router_prompt_path": "prompts/consumer_rights_router.yaml",
    # Adaptive feedback loop
    "adaptive_feedback_collection": "adaptive_feedback",
    "adaptive_feedback_log_collection": "adaptive_feedback_log",
    "adaptive_feedback_enabled": True,
    "adaptive_feedback_max_items": 5,
    "adaptive_critic_prompt_path": "prompts/adaptive_critic.yaml",
}


def _load_from_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def load_config() -> ComplianceConfig:
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_compliance_config.json")
    config_path = os.getenv("COMPLIANCE_CONFIG_PATH", default_path)
    file_data = _load_from_file(config_path)
    data = {**DEFAULT_CONFIG, **file_data}

    data["top_k_statutes"] = _to_int(os.getenv("COMPLIANCE_TOP_K_STATUTES"), data["top_k_statutes"])
    data["evidence_score_threshold"] = _to_float(
        os.getenv("COMPLIANCE_EVIDENCE_SCORE_THRESHOLD"), data["evidence_score_threshold"]
    )
    data["max_statute_chunk_chars"] = _to_int(
        os.getenv("COMPLIANCE_MAX_STATUTE_CHARS"), data["max_statute_chunk_chars"]
    )
    data["max_section_chars"] = _to_int(
        os.getenv("COMPLIANCE_MAX_SECTION_CHARS"), data["max_section_chars"]
    )
    data["use_embeddings_collection"] = _to_bool(
        os.getenv("COMPLIANCE_USE_EMBEDDINGS_COLLECTION"), data["use_embeddings_collection"]
    )
    data["cache_size"] = _to_int(os.getenv("COMPLIANCE_CACHE_SIZE"), data["cache_size"])
    data["cache_ttl_seconds"] = _to_int(
        os.getenv("COMPLIANCE_CACHE_TTL_SECONDS"), data["cache_ttl_seconds"]
    )
    data["llm_max_tokens"] = _to_int(os.getenv("COMPLIANCE_LLM_MAX_TOKENS"), data["llm_max_tokens"])
    data["llm_concurrency"] = _to_int(os.getenv("COMPLIANCE_LLM_CONCURRENCY"), data["llm_concurrency"])
    data["rate_limit_per_minute"] = _to_int(
        os.getenv("COMPLIANCE_RATE_LIMIT_PER_MINUTE"), data["rate_limit_per_minute"]
    )
    data["enable_redaction"] = _to_bool(
        os.getenv("COMPLIANCE_ENABLE_REDACTION"), data["enable_redaction"]
    )
    data["enable_audit_logging"] = _to_bool(
        os.getenv("COMPLIANCE_ENABLE_AUDIT_LOGGING"), data["enable_audit_logging"]
    )
    data["allow_raw_audit"] = _to_bool(
        os.getenv("COMPLIANCE_ALLOW_RAW_AUDIT"), data["allow_raw_audit"]
    )
    data["api_key"] = os.getenv("COMPLIANCE_API_KEY") or os.getenv("API_KEY", data["api_key"])
    # Auto-enable auth when an API key is configured (secure-by-default)
    auth_env = os.getenv("COMPLIANCE_AUTH_REQUIRED")
    if auth_env is not None:
        data["auth_required"] = _to_bool(auth_env, data["auth_required"])
    elif data["api_key"]:
        data["auth_required"] = True
    data["embedding_model_name"] = os.getenv("EMBEDDING_MODEL_NAME", data["embedding_model_name"])
    data["llm_model_name"] = os.getenv("LLM_MODEL_NAME", data["llm_model_name"])

    allowed_roles = os.getenv("COMPLIANCE_ALLOWED_ROLES")
    if allowed_roles:
        data["allowed_roles"] = [role.strip() for role in allowed_roles.split(",") if role.strip()]

    thresholds = data.get("confidence_thresholds", {})
    compliant_threshold = _to_float(
        os.getenv("COMPLIANCE_THRESHOLD_COMPLIANT"), thresholds.get("compliant", 0.75)
    )
    non_compliant_threshold = _to_float(
        os.getenv("COMPLIANCE_THRESHOLD_NON_COMPLIANT"), thresholds.get("non_compliant", 0.75)
    )

    confidence_thresholds = ConfidenceThresholds(
        compliant=compliant_threshold,
        non_compliant=non_compliant_threshold,
    )

    return ComplianceConfig(
        top_k_statutes=data["top_k_statutes"],
        confidence_thresholds=confidence_thresholds,
        evidence_score_threshold=data["evidence_score_threshold"],
        max_statute_chunk_chars=data["max_statute_chunk_chars"],
        max_section_chars=data["max_section_chars"],
        statutes_collection=data["statutes_collection"],
        embeddings_collection=data["embeddings_collection"],
        use_embeddings_collection=data["use_embeddings_collection"],
        vector_index_name=data["vector_index_name"],
        vector_field=data["vector_field"],
        statute_text_field=data["statute_text_field"],
        statute_title_field=data["statute_title_field"],
        statute_id_field=data["statute_id_field"],
        statute_section_id_field=data["statute_section_id_field"],
        statute_section_field=data.get("statute_section_field", DEFAULT_CONFIG["statute_section_field"]),
        statute_jurisdiction_field=data["statute_jurisdiction_field"],
        statute_corpus_field=data["statute_corpus_field"],
        embedding_collection_tag_field=data["embedding_collection_tag_field"],
        embedding_vector_field=data["embedding_vector_field"],
        embedding_text_field=data["embedding_text_field"],
        embedding_doc_id_field=data["embedding_doc_id_field"],
        embedding_chunk_id_field=data["embedding_chunk_id_field"],
        cache_size=data["cache_size"],
        cache_ttl_seconds=data["cache_ttl_seconds"],
        llm_max_tokens=data["llm_max_tokens"],
        llm_concurrency=data["llm_concurrency"],
        rate_limit_per_minute=data["rate_limit_per_minute"],
        audit_collection=data["audit_collection"],
        enable_redaction=data["enable_redaction"],
        enable_audit_logging=data["enable_audit_logging"],
        allow_raw_audit=data["allow_raw_audit"],
        auth_required=data["auth_required"],
        api_key=data["api_key"],
        allowed_roles=data["allowed_roles"],
        default_role_header=data["default_role_header"],
        embedding_model_name=data["embedding_model_name"],
        llm_model_name=data["llm_model_name"],
        statute_collection_tag=data["statute_collection_tag"],
        statute_database=(data.get("statute_database", "") or "").strip(),
        compliance_database=data.get("compliance_database", DEFAULT_CONFIG["compliance_database"]),
        policies_collection=data.get("policies_collection", DEFAULT_CONFIG["policies_collection"]),
        policy_chunks_collection=data.get("policy_chunks_collection", DEFAULT_CONFIG["policy_chunks_collection"]),
        policy_document_id_field=data.get("policy_document_id_field", DEFAULT_CONFIG["policy_document_id_field"]),
        policy_chunk_index_field=data.get("policy_chunk_index_field", DEFAULT_CONFIG["policy_chunk_index_field"]),
        policy_chunk_text_field=data.get("policy_chunk_text_field", DEFAULT_CONFIG["policy_chunk_text_field"]),
        policy_chunk_header_field=data.get("policy_chunk_header_field", DEFAULT_CONFIG["policy_chunk_header_field"]),
        statute_chunk_header_field=data.get("statute_chunk_header_field", DEFAULT_CONFIG["statute_chunk_header_field"]),
        compliance_results_collection=data.get("compliance_results_collection", DEFAULT_CONFIG["compliance_results_collection"]),
        compliance_alerts_collection=data.get("compliance_alerts_collection", DEFAULT_CONFIG["compliance_alerts_collection"]),
        compliance_run_log_collection=data.get("compliance_run_log_collection", DEFAULT_CONFIG["compliance_run_log_collection"]),
        compliance_jobs_collection=data.get("compliance_jobs_collection", DEFAULT_CONFIG["compliance_jobs_collection"]),
        index_jobs_collection=data.get("index_jobs_collection", DEFAULT_CONFIG["index_jobs_collection"]),
        statute_index_version_field=data.get("statute_index_version_field", DEFAULT_CONFIG["statute_index_version_field"]),
        max_statute_quote_chars=_to_int(
            os.getenv("COMPLIANCE_MAX_STATUTE_QUOTE_CHARS"),
            data.get("max_statute_quote_chars", DEFAULT_CONFIG["max_statute_quote_chars"]),
        ),
        default_jurisdictions=data.get("default_jurisdictions", DEFAULT_CONFIG["default_jurisdictions"]),
        conflict_penalty_multiplier=_to_float(
            os.getenv("COMPLIANCE_CONFLICT_PENALTY"), data.get("conflict_penalty_multiplier", DEFAULT_CONFIG["conflict_penalty_multiplier"])
        ),
        partial_credit_percent=_to_float(
            os.getenv("COMPLIANCE_PARTIAL_CREDIT_PERCENT"), data.get("partial_credit_percent", DEFAULT_CONFIG["partial_credit_percent"])
        ),
        requirement_weights=data.get("requirement_weights", DEFAULT_CONFIG["requirement_weights"]) or {},
        canonical_requirement_ids=data.get("canonical_requirement_ids", DEFAULT_CONFIG["canonical_requirement_ids"]),
        disclosure_queries=data.get("disclosure_queries", DEFAULT_CONFIG["disclosure_queries"]),
        statute_embeddings_collection=data.get("statute_embeddings_collection", DEFAULT_CONFIG["statute_embeddings_collection"]),
        policy_embeddings_collection=data.get("policy_embeddings_collection", DEFAULT_CONFIG["policy_embeddings_collection"]),
        gap_analysis_chunk_prompt_path=data.get("gap_analysis_chunk_prompt_path", DEFAULT_CONFIG["gap_analysis_chunk_prompt_path"]),
        statute_sub_embeddings_collection=data.get("statute_sub_embeddings_collection", DEFAULT_CONFIG["statute_sub_embeddings_collection"]),
        policy_sub_embeddings_collection=data.get("policy_sub_embeddings_collection", DEFAULT_CONFIG["policy_sub_embeddings_collection"]),
        statute_subchunk_text_field=data.get("statute_subchunk_text_field", DEFAULT_CONFIG["statute_subchunk_text_field"]),
        policy_subchunk_text_field=data.get("policy_subchunk_text_field", DEFAULT_CONFIG["policy_subchunk_text_field"]),
        statute_chunk_text_field=data.get("statute_chunk_text_field", DEFAULT_CONFIG["statute_chunk_text_field"]),
        gap_analysis_v3_top_k=_to_int(os.getenv("GAP_ANALYSIS_V3_TOP_K"), data.get("gap_analysis_v3_top_k", DEFAULT_CONFIG["gap_analysis_v3_top_k"])),
        gap_analysis_v3_num_candidates=_to_int(os.getenv("GAP_ANALYSIS_V3_NUM_CANDIDATES"), data.get("gap_analysis_v3_num_candidates", DEFAULT_CONFIG["gap_analysis_v3_num_candidates"])),
        gap_analysis_v3_score_threshold=_to_float(os.getenv("GAP_ANALYSIS_V3_SCORE_THRESHOLD"), data.get("gap_analysis_v3_score_threshold", DEFAULT_CONFIG["gap_analysis_v3_score_threshold"])),
        policy_embeddings_vector_index=data.get("policy_embeddings_vector_index", DEFAULT_CONFIG["policy_embeddings_vector_index"]),
        policy_sub_embeddings_vector_index=data.get("policy_sub_embeddings_vector_index", DEFAULT_CONFIG["policy_sub_embeddings_vector_index"]),
        gap_analysis_v3_prompt_path=data.get("gap_analysis_v3_prompt_path", DEFAULT_CONFIG["gap_analysis_v3_prompt_path"]),
        statute_sub_topic_embeddings_collection=data.get("statute_sub_topic_embeddings_collection", DEFAULT_CONFIG["statute_sub_topic_embeddings_collection"]),
        policy_legal_embeddings_collection=data.get("policy_legal_embeddings_collection", DEFAULT_CONFIG["policy_legal_embeddings_collection"]),
        category_mapping_collection=data.get("category_mapping_collection", DEFAULT_CONFIG["category_mapping_collection"]),
        gap_analysis_v4_prompt_path=data.get("gap_analysis_v4_prompt_path", DEFAULT_CONFIG["gap_analysis_v4_prompt_path"]),
        consumer_rights_router_prompt_path=data.get("consumer_rights_router_prompt_path", DEFAULT_CONFIG["consumer_rights_router_prompt_path"]),
        adaptive_feedback_collection=data.get("adaptive_feedback_collection", DEFAULT_CONFIG["adaptive_feedback_collection"]),
        adaptive_feedback_log_collection=data.get("adaptive_feedback_log_collection", DEFAULT_CONFIG["adaptive_feedback_log_collection"]),
        adaptive_feedback_enabled=_to_bool(os.getenv("ADAPTIVE_FEEDBACK_ENABLED"), data.get("adaptive_feedback_enabled", DEFAULT_CONFIG["adaptive_feedback_enabled"])),
        adaptive_feedback_max_items=_to_int(os.getenv("ADAPTIVE_FEEDBACK_MAX_ITEMS"), data.get("adaptive_feedback_max_items", DEFAULT_CONFIG["adaptive_feedback_max_items"])),
        adaptive_critic_prompt_path=data.get("adaptive_critic_prompt_path", DEFAULT_CONFIG["adaptive_critic_prompt_path"]),
    )
