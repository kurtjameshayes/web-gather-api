"""Regression tests for adaptive-feedback config wiring and injection limits.

``ADAPTIVE_FEEDBACK_ENABLED`` is coerced only when the env var is set. An
empty env value disables feedback. A file string ``"false"`` is stored as
that string and stays truthy. ``ADAPTIVE_FEEDBACK_MAX_ITEMS`` of ``0`` or a
negative integer is forwarded to Mongo ``limit`` unchanged. ``limit(0)`` is
unlimited, unlike the HTTP route which floors at 1.
"""
from __future__ import annotations

import json

from unittest.mock import MagicMock

from adaptive_feedback_service import CriticService
from compliance_config import load_config


_FEEDBACK_ENV = (
    "ADAPTIVE_FEEDBACK_ENABLED",
    "ADAPTIVE_FEEDBACK_MAX_ITEMS",
    "COMPLIANCE_CONFIG_PATH",
)


def _clear_feedback_env(monkeypatch) -> None:
    for name in _FEEDBACK_ENV:
        monkeypatch.delenv(name, raising=False)


def _limit_cursor() -> MagicMock:
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = []
    return cursor


def test_feedback_flags_from_env_and_file(monkeypatch, tmp_path) -> None:
    _clear_feedback_env(monkeypatch)
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is True
    assert cfg.adaptive_feedback_max_items == 5

    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "2.5")
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is False
    assert cfg.adaptive_feedback_max_items == 5

    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "  YES  ")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "0")
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is True
    assert cfg.adaptive_feedback_max_items == 0

    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "off")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "  4  ")
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is False
    assert cfg.adaptive_feedback_max_items == 4

    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "-3")
    cfg = load_config()
    assert cfg.adaptive_feedback_max_items == -3

    _clear_feedback_env(monkeypatch)
    config_path = tmp_path / "compliance.json"
    config_path.write_text(
        json.dumps(
            {
                "adaptive_feedback_enabled": False,
                "adaptive_feedback_max_items": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COMPLIANCE_CONFIG_PATH", str(config_path))
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is False
    assert cfg.adaptive_feedback_max_items == 2

    monkeypatch.setenv("ADAPTIVE_FEEDBACK_ENABLED", "true")
    monkeypatch.setenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", "9")
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled is True
    assert cfg.adaptive_feedback_max_items == 9

    monkeypatch.delenv("ADAPTIVE_FEEDBACK_ENABLED", raising=False)
    monkeypatch.delenv("ADAPTIVE_FEEDBACK_MAX_ITEMS", raising=False)
    config_path.write_text(
        json.dumps({"adaptive_feedback_enabled": "false"}),
        encoding="utf-8",
    )
    cfg = load_config()
    assert cfg.adaptive_feedback_enabled == "false"
    assert bool(cfg.adaptive_feedback_enabled) is True
    assert cfg.adaptive_feedback_max_items == 5


def test_active_feedback_forwards_zero_and_negative_limits() -> None:
    cfg = load_config()
    cfg.adaptive_feedback_collection = "adaptive_feedback"
    cfg.compliance_database = "privacy-compliance"
    mongo = MagicMock()
    coll = MagicMock()
    cursor = _limit_cursor()
    coll.find.return_value = cursor
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    critic = CriticService(mongo_client=mongo, config=cfg, llm_client=MagicMock())

    cfg.adaptive_feedback_max_items = 0
    critic.get_active_feedback("pol-1")
    coll.find.assert_called_with(
        {"policy_document_id": "pol-1", "superseded_by": None}
    )
    cursor.limit.assert_called_with(0)

    cursor.limit.reset_mock()
    critic.get_active_feedback("pol-1", limit=0)
    cursor.limit.assert_called_with(0)

    cursor.limit.reset_mock()
    critic.get_active_feedback("pol-1", limit=2)
    cursor.limit.assert_called_with(2)

    cfg.adaptive_feedback_max_items = -3
    cursor.limit.reset_mock()
    critic.get_active_feedback("pol-1")
    cursor.limit.assert_called_with(-3)
