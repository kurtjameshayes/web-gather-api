"""Jurisdiction default used by statute vector search.

vector_retriever reads COMPLIANCE_VECTOR_SEARCH_JURISDICTION (strip + empty
fallback to California). Gap analysis v3 hardcodes the same default and
ignores the env var. PR #209 covers empty-jurisdiction *filter fallback* using
the already-loaded constant, not env loading or the v3/retriever split.
"""
from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import gap_analysis_service_v3
import vector_retriever


def _reload_retriever_with_env(monkeypatch, value: str | None) -> str:
    if value is None:
        monkeypatch.delenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", raising=False)
    else:
        monkeypatch.setenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", value)
    module = importlib.reload(vector_retriever)
    return module.VECTOR_SEARCH_JURISDICTION


def test_vector_retriever_jurisdiction_env_override_and_whitespace_fallback(
    monkeypatch,
) -> None:
    """Unset/blank env falls back to California; non-empty values are stripped."""
    original = os.getenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION")
    try:
        assert _reload_retriever_with_env(monkeypatch, None) == "California"
        assert _reload_retriever_with_env(monkeypatch, "") == "California"
        assert _reload_retriever_with_env(monkeypatch, "   ") == "California"
        assert _reload_retriever_with_env(monkeypatch, "  New York  ") == "New York"
        assert _reload_retriever_with_env(monkeypatch, "CA") == "CA"
    finally:
        if original is None:
            monkeypatch.delenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", raising=False)
        else:
            monkeypatch.setenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", original)
        importlib.reload(vector_retriever)


def test_gap_analysis_v3_jurisdiction_constant_ignores_env(monkeypatch) -> None:
    """V3 keeps a hardcoded California default even when the retriever env is set."""
    monkeypatch.setenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", "New York")
    reloaded = importlib.reload(gap_analysis_service_v3)
    assert reloaded.VECTOR_SEARCH_JURISDICTION == "California"
    monkeypatch.delenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION", raising=False)
    importlib.reload(gap_analysis_service_v3)
