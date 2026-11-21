"""Regression tests for compliance config precedence.

The checked-in file switches retrieval onto statute_embeddings, and an API key
turns authentication on unless COMPLIANCE_AUTH_REQUIRED explicitly disables it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance_config import load_config

_CONFIG_ENV_VARS = (
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


@pytest.fixture
def isolated_config_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write_config(directory: Path, payload: dict) -> str:
    path = directory / "policy_compliance_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_checked_in_config_selects_statute_embeddings(
    isolated_config_env: pytest.MonkeyPatch,
) -> None:
    config = load_config()

    assert config.use_embeddings_collection is True
    assert config.embeddings_collection == "statute_embeddings"
    assert config.embedding_vector_field == "embedding"
    assert config.embedding_text_field == "chunk_text"
    assert config.embedding_doc_id_field == "document_id"
    assert config.statute_collection_tag == ""
    assert config.statute_database == ""
    assert config.auth_required is False
    assert config.api_key == ""
    assert config.allowed_roles == ["admin", "compliance"]
    assert config.adaptive_feedback_enabled is True
    assert config.requirement_weights["right to delete"] == 1.5


@pytest.mark.parametrize("contents", ["", "{", "not-json"])
def test_unreadable_config_file_keeps_code_defaults(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "policy_compliance_config.json"
    if contents:
        path.write_text(contents, encoding="utf-8")
    else:
        path = tmp_path / "missing.json"
    isolated_config_env.setenv("COMPLIANCE_CONFIG_PATH", str(path))

    config = load_config()

    assert config.use_embeddings_collection is False
    assert config.embeddings_collection == "embeddings"
    assert config.embedding_vector_field == "vector"
    assert config.top_k_statutes == 5
    assert config.auth_required is False
    assert config.api_key == ""
    assert config.enable_redaction is True
    assert config.allowed_roles == ["admin", "compliance"]
    assert config.requirement_weights["right to delete"] == 1.5
    assert config.conflict_penalty_multiplier == 0.7
    assert config.partial_credit_percent == 0.5


def test_file_replaces_requirement_weights_instead_of_merging(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"requirement_weights": {"only rule": 4.5}}),
    )

    config = load_config()

    assert config.requirement_weights == {"only rule": 4.5}


def test_null_requirement_weights_become_an_empty_map(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"requirement_weights": None}),
    )

    config = load_config()

    assert config.requirement_weights == {}


def test_env_overrides_file_and_invalid_numbers_keep_the_file_value(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(
            tmp_path,
            {
                "top_k_statutes": 11,
                "evidence_score_threshold": 0.4,
                "use_embeddings_collection": True,
                "enable_redaction": False,
                "enable_audit_logging": True,
                "allow_raw_audit": False,
                "confidence_thresholds": {"compliant": 0.5, "non_compliant": 0.4},
                "embedding_model_name": "file-model",
                "llm_model_name": "file-llm",
                "statute_database": "  ca-db  ",
                "partial_credit_percent": 0.25,
                "conflict_penalty_multiplier": 0.8,
                "gap_analysis_v3_score_threshold": 0.7,
                "adaptive_feedback_enabled": True,
                "adaptive_feedback_max_items": 5,
            },
        ),
    )
    isolated_config_env.setenv("COMPLIANCE_TOP_K_STATUTES", " 8 ")
    isolated_config_env.setenv("COMPLIANCE_EVIDENCE_SCORE_THRESHOLD", "nope")
    isolated_config_env.setenv("COMPLIANCE_USE_EMBEDDINGS_COLLECTION", "0")
    isolated_config_env.setenv("COMPLIANCE_ENABLE_REDACTION", " YeS ")
    isolated_config_env.setenv("COMPLIANCE_ENABLE_AUDIT_LOGGING", "off")
    isolated_config_env.setenv("COMPLIANCE_ALLOW_RAW_AUDIT", "2")
    isolated_config_env.setenv("COMPLIANCE_THRESHOLD_COMPLIANT", "0.9")
    isolated_config_env.setenv("COMPLIANCE_THRESHOLD_NON_COMPLIANT", "bad")
    isolated_config_env.setenv("EMBEDDING_MODEL_NAME", "env-model")
    isolated_config_env.setenv("LLM_MODEL_NAME", "env-llm")
    isolated_config_env.setenv("COMPLIANCE_PARTIAL_CREDIT_PERCENT", "-1")
    isolated_config_env.setenv("COMPLIANCE_CONFLICT_PENALTY", "")
    isolated_config_env.setenv("GAP_ANALYSIS_V3_SCORE_THRESHOLD", "1.5")
    isolated_config_env.setenv("ADAPTIVE_FEEDBACK_ENABLED", "no")
    isolated_config_env.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "0")

    config = load_config()

    assert config.top_k_statutes == 8
    assert config.evidence_score_threshold == 0.4
    assert config.use_embeddings_collection is False
    assert config.enable_redaction is True
    assert config.enable_audit_logging is False
    assert config.allow_raw_audit is False
    assert config.confidence_thresholds.compliant == 0.9
    assert config.confidence_thresholds.non_compliant == 0.4
    assert config.embedding_model_name == "env-model"
    assert config.llm_model_name == "env-llm"
    assert config.statute_database == "ca-db"
    assert config.partial_credit_percent == -1.0
    assert config.conflict_penalty_multiplier == 0.8
    assert config.gap_analysis_v3_score_threshold == 1.5
    assert config.adaptive_feedback_enabled is False
    assert config.adaptive_feedback_max_items == 0


def test_invalid_top_k_keeps_the_file_value(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"top_k_statutes": 11}),
    )
    isolated_config_env.setenv("COMPLIANCE_TOP_K_STATUTES", "nope")

    assert load_config().top_k_statutes == 11


def test_file_api_key_enables_auth_without_an_auth_env_flag(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"api_key": "from-file", "auth_required": False}),
    )

    config = load_config()

    assert config.api_key == "from-file"
    assert config.auth_required is True


def test_compliance_api_key_enables_auth(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"auth_required": False}),
    )
    isolated_config_env.setenv("COMPLIANCE_API_KEY", "secret")

    config = load_config()

    assert config.api_key == "secret"
    assert config.auth_required is True


@pytest.mark.parametrize("flag", ["false", "0", "off", "no"])
def test_explicit_auth_flag_can_stay_off_while_a_key_is_set(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
    flag: str,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {}),
    )
    isolated_config_env.setenv("COMPLIANCE_API_KEY", "secret")
    isolated_config_env.setenv("COMPLIANCE_AUTH_REQUIRED", flag)

    config = load_config()

    assert config.api_key == "secret"
    assert config.auth_required is False


def test_whitespace_api_key_is_kept_and_still_enables_auth(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"auth_required": False}),
    )
    isolated_config_env.setenv("COMPLIANCE_API_KEY", "  ")

    config = load_config()

    assert config.api_key == "  "
    assert config.auth_required is True


def test_empty_compliance_api_key_falls_through_to_api_key(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"api_key": "from-file", "auth_required": False}),
    )
    isolated_config_env.setenv("COMPLIANCE_API_KEY", "")
    isolated_config_env.setenv("API_KEY", "legacy")

    config = load_config()

    assert config.api_key == "legacy"
    assert config.auth_required is True


def test_auth_required_true_without_a_key_stays_enabled(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"auth_required": True}),
    )
    isolated_config_env.setenv("COMPLIANCE_AUTH_REQUIRED", "true")

    config = load_config()

    assert config.api_key == ""
    assert config.auth_required is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" admin , , compliance , ", ["admin", "compliance"]),
        (" , ", []),
        ("", None),
    ],
)
def test_allowed_roles_split_drops_blanks_and_empty_env_keeps_the_file(
    isolated_config_env: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: str,
    expected: list[str] | None,
) -> None:
    isolated_config_env.setenv(
        "COMPLIANCE_CONFIG_PATH",
        _write_config(tmp_path, {"allowed_roles": ["auditor"]}),
    )
    isolated_config_env.setenv("COMPLIANCE_ALLOWED_ROLES", raw)

    config = load_config()

    assert config.allowed_roles == (["auditor"] if expected is None else expected)
