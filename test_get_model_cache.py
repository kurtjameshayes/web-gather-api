"""Regression tests for core.get_model cache bounds and allowlist."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import core


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    """Keep get_model cache isolated across tests."""
    core._model_cache.clear()
    yield
    core._model_cache.clear()


def _fill_cache(n: int = core._MAX_MODEL_CACHE_SIZE) -> list[str]:
    names = [f"unlisted-model-{i}" for i in range(n)]
    with patch("core.SentenceTransformer", side_effect=lambda name: f"loaded:{name}") as ctor:
        for name in names:
            assert core.get_model(name) == f"loaded:{name}"
        assert ctor.call_count == n
    return names


def test_get_model_returns_cached_instance_without_reloading() -> None:
    """A second lookup for the same name must reuse the cached model."""
    with patch("core.SentenceTransformer", return_value="model-a") as ctor:
        first = core.get_model("custom-a")
        second = core.get_model("custom-a")
    assert first == "model-a"
    assert second is first
    ctor.assert_called_once_with("custom-a")


def test_get_model_rejects_unlisted_name_when_cache_is_full() -> None:
    """Unknown model names must not grow the cache past the bounded size."""
    _fill_cache()
    with patch("core.SentenceTransformer") as ctor:
        with pytest.raises(ValueError, match="not in the allowlist"):
            core.get_model("attacker-controlled-model")
    ctor.assert_not_called()
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE


def test_get_model_allowlisted_name_evicts_oldest_when_cache_is_full() -> None:
    """Allowlisted models can still load by evicting the oldest cache entry."""
    names = _fill_cache()
    allowed = next(iter(core.ALLOWED_EMBEDDING_MODELS))
    with patch("core.SentenceTransformer", return_value="allowed-model") as ctor:
        loaded = core.get_model(allowed)
    assert loaded == "allowed-model"
    ctor.assert_called_once_with(allowed)
    assert allowed in core._model_cache
    assert names[0] not in core._model_cache
    assert len(core._model_cache) == core._MAX_MODEL_CACHE_SIZE


def test_get_model_cache_hit_bypasses_allowlist_when_full() -> None:
    """Already-cached models remain retrievable after the cache fills."""
    names = _fill_cache()
    with patch("core.SentenceTransformer") as ctor:
        cached = core.get_model(names[0])
    assert cached == f"loaded:{names[0]}"
    ctor.assert_not_called()
