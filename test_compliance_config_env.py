"""Regression tests for compliance config env and file overrides.

Auth turns on whenever an API key is present unless COMPLIANCE_AUTH_REQUIRED
explicitly disables it. Numeric garbage falls back to the file or the
built-in default. A config file replaces mapping fields instead of merging
them, and string booleans in that file are not parsed unless they come from
the environment.
"""
from __future__ import annotations

import json

import pytest

from compliance_config import load_config

_ENV_KEYS = (
    "COMPLIANCE_CONFIG_PATH",
    "COMPLIANCE_TOP_K_STATUTES",
    "COMPLIANCE_EVIDENCE_SCORE_THRESHOLD",
    "COMPLIANCE_MAX_STATUTE_CHARS",
    "COMPLIANCE_MAX_SECTION_CHARS",
    "COMPLIANCE_USE_EMBEDDINGS_COLLECTION",
    "COMPLIANCE_CACHE_SIZE",
    "COMPLIANCE_CACHE_TTL_SECONDS",
    "COMPLIANCE_LLM_MAX_TOKENS",
    "COMPLIANCE_LLM_CONCURRENCY",
    "COMPLIANCE_RATE_LIMIT_PER_MINUTE",
    "COMPLIANCE_ENABLE_REDACTION",
    "COMPLIANCE_ENABLE_AUDIT_LOGGING",
    "COMPLIANCE_ALLOW_RAW_AUDIT",
    "COMPLIANCE_API_KEY",
    "API_KEY",
    "COMPLIANCE_AUTH_REQUIRED",
    "EMBEDDING_MODEL_NAME",
    "LLM_MODEL_NAME",
    "COMPLIANCE_ALLOWED_ROLES",
    "COMPLIANCE_THRESHOLD_COMPLIANT",
    "COMPLIANCE_THRESHOLD_NON_COMPLIANT",
    "COMPLIANCE_MAX_STATUTE_QUOTE_CHARS",
    "COMPLIANCE_CONFLICT_PENALTY",
    "COMPLIANCE_PARTIAL_CREDIT_PERCENT",
    "GAP_ANALYSIS_V3_TOP_K",
    "GAP_ANALYSIS_V3_NUM_CANDIDATES",
    "GAP_ANALYSIS_V3_SCORE_THRESHOLD",
    "ADAPTIVE_FEEDBACK_ENABLED",
    "ADAPTIVE_FEEDBACK_MAX_ITEMS",
)


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path, payload=None, raw=None) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if raw is not None:
        path = tmp_path / "cfg.json"
        path.write_text(raw, encoding="utf-8")
        monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(path))
    elif payload is not None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(path))
    else:
        monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(tmp_path / "missing.json"))


def test_missing_file_uses_defaults_and_leaves_auth_off(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)
    cfg = load_config()
    assert cfg.auth_required is False
    assert cfg.api_key == ""
    assert cfg.top_k_statutes == 5
    assert cfg.evidence_score_threshold == 0.2
    assert cfg.conflict_penalty_multiplier == 0.7
    assert cfg.partial_credit_percent == 0.5
    assert cfg.max_statute_quote_chars == 300
    assert cfg.gap_analysis_v3_score_threshold == 0.70
    assert cfg.use_embeddings_collection is False
    assert cfg.embedding_vector_field == "vector"
    assert cfg.adaptive_feedback_enabled is True
    assert cfg.statute_database == ""
    assert cfg.default_jurisdictions == ["CA", "VA", "CO", "CT"]
    assert cfg.requirement_weights["right to delete"] == 1.5
    assert cfg.allowed_roles == ["admin", "compliance"]


def test_api_key_enables_auth_unless_explicitly_disabled(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("COMPLIANCE_API_KEY", "primary")
    monkeypatch.setenv("API_KEY", "secondary")
    cfg = load_config()
    assert cfg.api_key == "primary"
    assert cfg.auth_required is True

    monkeypatch.setenv("COMPLIANCE_API_KEY", "")
    cfg = load_config()
    assert cfg.api_key == "secondary"
    assert cfg.auth_required is True

    monkeypatch.setenv("COMPLIANCE_API_KEY", "  spaced  ")
    cfg = load_config()
    assert cfg.api_key == "  spaced  "
    assert cfg.auth_required is True

    for value in ("false", "off", "0", "", "no", "  FaLsE  "):
        monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", value)
        assert load_config().auth_required is False

    monkeypatch.delenv("COMPLIANCE_API_KEY")
    monkeypatch.delenv("API_KEY")
    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "  YES ")
    cfg = load_config()
    assert cfg.api_key == ""
    assert cfg.auth_required is True

    monkeypatch.setenv("COMPLIANCE_AUTH_REQUIRED", "on")
    assert load_config().auth_required is True


def test_invalid_numeric_env_falls_back_and_whitespace_parses(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("COMPLIANCE_TOP_K_STATUTES", "nope")
    monkeypatch.setenv("COMPLIANCE_MAX_SECTION_CHARS", "10.5")
    monkeypatch.setenv("COMPLIANCE_CONFLICT_PENALTY", "nope")
    monkeypatch.setenv("COMPLIANCE_PARTIAL_CREDIT_PERCENT", "")
    monkeypatch.setenv("GAP_ANALYSIS_V3_SCORE_THRESHOLD", "abc")
    monkeypatch.setenv("COMPLIANCE_MAX_STATUTE_QUOTE_CHARS", "  40")
    monkeypatch.setenv("COMPLIANCE_EVIDENCE_SCORE_THRESHOLD", "  0.25 ")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "  2")
    cfg = load_config()
    assert cfg.top_k_statutes == 5
    assert cfg.max_section_chars == 2000
    assert cfg.conflict_penalty_multiplier == 0.7
    assert cfg.partial_credit_percent == 0.5
    assert cfg.gap_analysis_v3_score_threshold == 0.70
    assert cfg.max_statute_quote_chars == 40
    assert cfg.evidence_score_threshold == 0.25
    assert cfg.adaptive_feedback_max_items == 2


def test_file_replaces_mappings_and_string_bools_stay_unparsed(monkeypatch, tmp_path) -> None:
    _isolate(
        monkeypatch,
        tmp_path,
        payload={
            "auth_required": False,
            "api_key": "from-file",
            "requirement_weights": {"custom rule": 9},
            "requirement_weights_ignored": True,
            "statute_database": "  statutes-db  ",
            "confidence_thresholds": {"compliant": 0.9},
            "max_statute_quote_chars": 120,
            "conflict_penalty_multiplier": 0.2,
            "use_embeddings_collection": "yes",
            "adaptive_feedback_enabled": "false",
            "embedding_vector_field": "embedding",
            "canonical_requirement_ids": ["only_one"],
        },
    )
    cfg = load_config()
    assert cfg.auth_required is True
    assert cfg.api_key == "from-file"
    assert cfg.requirement_weights == {"custom rule": 9}
    assert cfg.statute_database == "statutes-db"
    assert cfg.confidence_thresholds.compliant == 0.9
    assert cfg.confidence_thresholds.non_compliant == 0.75
    assert cfg.max_statute_quote_chars == 120
    assert cfg.conflict_penalty_multiplier == 0.2
    assert cfg.use_embeddings_collection == "yes"
    assert cfg.adaptive_feedback_enabled == "false"
    assert cfg.embedding_vector_field == "embedding"
    assert cfg.canonical_requirement_ids == ["only_one"]

    monkeypatch.setenv("COMPLIANCE_CONFLICT_PENALTY", "nope")
    monkeypatch.setenv("COMPLIANCE_USE_EMBEDDINGS_COLLECTION", "no")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "off")
    monkeypatch.setenv("COMPLIANCE_THRESHOLD_NON_COMPLIANT", "0.4")
    cfg = load_config()
    assert cfg.conflict_penalty_multiplier == 0.2
    assert cfg.use_embeddings_collection is False
    assert cfg.adaptive_feedback_enabled is False
    assert cfg.confidence_thresholds.non_compliant == 0.4

    _isolate(
        monkeypatch,
        tmp_path,
        payload={"requirement_weights": None, "statute_database": None, "auth_required": True},
    )
    cfg = load_config()
    assert cfg.requirement_weights == {}
    assert cfg.statute_database == ""
    assert cfg.auth_required is True
    assert cfg.api_key == ""

    _isolate(monkeypatch, tmp_path, payload={"auth_required": False})
    cfg = load_config()
    assert cfg.auth_required is False
    assert cfg.api_key == ""


def test_invalid_json_file_is_ignored(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path, raw="{not json")
    cfg = load_config()
    assert cfg.auth_required is False
    assert cfg.top_k_statutes == 5
    assert cfg.requirement_weights["right to know"] == 1.5
