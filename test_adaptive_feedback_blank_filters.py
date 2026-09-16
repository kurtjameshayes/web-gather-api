"""Adaptive-feedback blank query filters: history vs log.

History: empty policy_document_id is 400; whitespace is truthy and forwarded.
Log: empty run_id is falsy in the helper (unfiltered); whitespace is a real
run_id filter. Distinct from #230 (omitted run_id=None) and #236 (offset/auth).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("pyppeteer", MagicMock())

import compliance_routes
from adaptive_feedback_service import CriticService
from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp


@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    return config


@pytest.fixture
def client(mock_config):
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    return app.test_client()


def test_list_feedback_empty_policy_id_is_400_whitespace_is_forwarded(
    client, mock_config
) -> None:
    mock_critic = MagicMock()
    mock_critic.get_feedback_history.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        empty = client.get("/api/v4/compliance/adaptive-feedback?policy_document_id=")
        assert empty.status_code == 400
        assert "policy_document_id" in empty.get_json()["error"]
        mock_critic.get_feedback_history.assert_not_called()

        whitespace = client.get(
            "/api/v4/compliance/adaptive-feedback?policy_document_id=%20%20%20"
        )
    assert whitespace.status_code == 200
    mock_critic.get_feedback_history.assert_called_once_with(
        policy_document_id="   ",
        active_only=False,
        limit=50,
        offset=0,
    )


def test_list_feedback_log_empty_vs_whitespace_run_id(client, mock_config) -> None:
    mock_critic = MagicMock()
    mock_critic.get_feedback_log.return_value = []
    with patch.object(compliance_routes, "_config", mock_config), patch.object(
        compliance_routes, "_critic_service", mock_critic
    ):
        empty = client.get("/api/v4/compliance/adaptive-feedback/log?run_id=")
        assert empty.status_code == 200
        mock_critic.get_feedback_log.assert_called_with(run_id="", limit=50, offset=0)

        mock_critic.get_feedback_log.reset_mock()
        whitespace = client.get("/api/v4/compliance/adaptive-feedback/log?run_id=%20%20")
    assert whitespace.status_code == 200
    mock_critic.get_feedback_log.assert_called_once_with(run_id="  ", limit=50, offset=0)


def test_get_feedback_log_empty_is_unfiltered_whitespace_filters() -> None:
    """Helper: empty/None skip the run_id clause; whitespace becomes an equality filter."""
    mock_mongo = MagicMock()
    coll = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    coll.find.return_value = cursor
    mock_mongo.__getitem__.return_value.__getitem__.return_value = coll
    config = load_config()
    critic = CriticService(mongo_client=mock_mongo, config=config, llm_client=MagicMock())

    critic.get_feedback_log(run_id="")
    coll.find.assert_called_with({})

    coll.find.reset_mock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    coll.find.return_value = cursor
    critic.get_feedback_log(run_id="   ")
    coll.find.assert_called_with({"run_id": "   "})
