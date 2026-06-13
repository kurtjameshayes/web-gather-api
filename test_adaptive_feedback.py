"""Tests for adaptive feedback service, CriticService, and V4 endpoints."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from compliance_routes import compliance_bp
from compliance_routes_v4 import compliance_v4_bp
from compliance_config import load_config
from compliance_suite_schemas import (
    GapAnalysisResponse,
    GapItem,
    GapSummary,
    RetrievalMetadata,
)
from adaptive_feedback_service import CriticService, VALID_CATEGORIES, VALID_SEVERITIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    config = load_config()
    config.auth_required = False
    config.adaptive_feedback_enabled = True
    config.adaptive_feedback_max_items = 5
    config.adaptive_feedback_collection = "adaptive_feedback"
    config.adaptive_feedback_log_collection = "adaptive_feedback_log"
    config.adaptive_critic_prompt_path = "prompts/adaptive_critic.yaml"
    return config


@pytest.fixture
def mock_mongo():
    return MagicMock()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def critic(mock_mongo, mock_config, mock_llm):
    return CriticService(
        mongo_client=mock_mongo,
        config=mock_config,
        llm_client=mock_llm,
    )


@pytest.fixture
def sample_response():
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-03-18T00:00:00Z",
        gaps=[
            GapItem(
                jurisdiction="CA",
                statute_reference="§ 1798.100",
                requirement_summary="Right to know",
                status="addressed",
                policy_quote="We collect personal information.",
                confidence="high",
            ),
            GapItem(
                jurisdiction="CA",
                statute_reference="§ 1798.105",
                requirement_summary="Right to delete",
                status="missing",
                conflict_description="Policy does not address deletion.",
                confidence="high",
            ),
        ],
        summary=GapSummary(
            total_requirements=2,
            missing=1,
            addressed=1,
            conflicts=0,
        ),
        retrieval_metadata=RetrievalMetadata(
            statute_items_considered=2,
            statute_pairs_matched=2,
        ),
        run_types=["gap_v4"],
        run_type="gap_analysis_v4",
        version="v4",
    )


# ---------------------------------------------------------------------------
# CriticService._validate_suggestions
# ---------------------------------------------------------------------------

class TestValidateSuggestions:
    def test_valid_suggestions(self):
        raw = [
            {
                "category": "false_positive",
                "description": "Some desc",
                "instruction": "Do X instead of Y",
                "statute_reference": "§ 1798.100",
                "severity": "high",
            },
            {
                "category": "prompt_guidance",
                "description": "General tip",
                "instruction": "Always check for Z",
                "severity": "low",
            },
        ]
        result = CriticService._validate_suggestions(raw)
        assert len(result) == 2
        assert result[0]["category"] == "false_positive"
        assert result[1]["statute_reference"] is None

    def test_filters_invalid_category(self):
        raw = [{"category": "invalid_cat", "instruction": "Do stuff", "severity": "high"}]
        result = CriticService._validate_suggestions(raw)
        assert len(result) == 0

    def test_filters_missing_instruction(self):
        raw = [{"category": "false_positive", "instruction": "", "severity": "high"}]
        result = CriticService._validate_suggestions(raw)
        assert len(result) == 0

    def test_caps_at_10(self):
        raw = [
            {"category": "prompt_guidance", "instruction": f"Tip {i}", "severity": "low"}
            for i in range(15)
        ]
        result = CriticService._validate_suggestions(raw)
        assert len(result) == 10

    def test_normalizes_severity(self):
        raw = [{"category": "false_negative", "instruction": "Do X", "severity": "EXTREME"}]
        result = CriticService._validate_suggestions(raw)
        assert result[0]["severity"] == "medium"

    def test_non_list_returns_empty(self):
        assert CriticService._validate_suggestions("not a list") == []
        assert CriticService._validate_suggestions(None) == []


# ---------------------------------------------------------------------------
# CriticService.format_feedback_for_prompt
# ---------------------------------------------------------------------------

class TestFormatFeedback:
    def test_formats_instructions(self, critic):
        docs = [
            {
                "_id": "fb-1",
                "suggestions": [
                    {"instruction": "Check for partial compliance"},
                    {"instruction": "Be strict with citations"},
                ],
            },
            {
                "_id": "fb-2",
                "suggestions": [
                    {"instruction": "Watch for data retention gaps"},
                ],
            },
        ]
        result = critic.format_feedback_for_prompt(docs)
        assert "PRIOR ANALYSIS FEEDBACK" in result
        assert "- Check for partial compliance" in result
        assert "- Be strict with citations" in result
        assert "- Watch for data retention gaps" in result

    def test_empty_suggestions(self, critic):
        docs = [{"_id": "fb-1", "suggestions": []}]
        assert critic.format_feedback_for_prompt(docs) == ""

    def test_empty_docs(self, critic):
        assert critic.format_feedback_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# CriticService.get_active_feedback
# ---------------------------------------------------------------------------

class TestGetActiveFeedback:
    def test_queries_for_non_superseded(self, critic, mock_mongo):
        coll_mock = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.sort.return_value = cursor_mock
        cursor_mock.limit.return_value = [
            {"_id": "fb-1", "suggestions": [], "superseded_by": None}
        ]
        coll_mock.find.return_value = cursor_mock
        mock_mongo.__getitem__.return_value.__getitem__.return_value = coll_mock

        result = critic.get_active_feedback("pol-1")
        assert len(result) == 1
        assert result[0]["_id"] == "fb-1"
        coll_mock.find.assert_called_once_with(
            {"policy_document_id": "pol-1", "superseded_by": None}
        )


# ---------------------------------------------------------------------------
# CriticService.evaluate (async)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_stores_feedback(self, critic, mock_mongo, sample_response):
        coll_mock = MagicMock()
        mock_mongo.__getitem__.return_value.__getitem__.return_value = coll_mock

        critic._llm._call_json = AsyncMock(return_value={
            "summary_evaluation": "Good overall analysis.",
            "suggestions": [
                {
                    "category": "false_positive",
                    "description": "Citation too vague",
                    "instruction": "Require exact verbatim quotes",
                    "statute_reference": "§ 1798.100",
                    "severity": "high",
                },
            ],
        })

        with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
            feedback_id = _run_async(critic.evaluate(sample_response, "run-123"))

        assert feedback_id is not None
        coll_mock.insert_one.assert_called_once()
        inserted = coll_mock.insert_one.call_args[0][0]
        assert inserted["run_id"] == "run-123"
        assert inserted["policy_document_id"] == "pol-1"
        assert len(inserted["suggestions"]) == 1
        assert inserted["gap_summary_snapshot"]["total_requirements"] == 2

    def test_evaluate_disabled(self, critic, mock_config, sample_response):
        mock_config.adaptive_feedback_enabled = False
        critic._cfg = mock_config
        result = _run_async(critic.evaluate(sample_response, "run-123"))
        assert result is None

    def test_evaluate_handles_llm_failure(self, critic, mock_mongo, sample_response):
        coll_mock = MagicMock()
        mock_mongo.__getitem__.return_value.__getitem__.return_value = coll_mock

        critic._llm._call_json = AsyncMock(return_value=None)

        result = _run_async(critic.evaluate(sample_response, "run-123"))
        assert result is None
        coll_mock.insert_one.assert_not_called()


# ---------------------------------------------------------------------------
# CriticService.record_feedback_usage (async)
# ---------------------------------------------------------------------------

class TestRecordFeedbackUsage:
    def test_records_usage(self, critic, mock_mongo):
        coll_mock = MagicMock()
        mock_mongo.__getitem__.return_value.__getitem__.return_value = coll_mock

        with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
            _run_async(
                critic.record_feedback_usage(
                    run_id="run-456",
                    feedback_ids=["fb-1", "fb-2"],
                    rendered_text="- Do X\n- Do Y",
                )
            )

        coll_mock.insert_one.assert_called_once()
        inserted = coll_mock.insert_one.call_args[0][0]
        assert inserted["run_id"] == "run-456"
        assert inserted["feedback_ids_used"] == ["fb-1", "fb-2"]
        assert inserted["feedback_instructions_text"] == "- Do X\n- Do Y"


# ---------------------------------------------------------------------------
# V4 endpoint: GET /adaptive-feedback
# ---------------------------------------------------------------------------

class TestAdaptiveFeedbackEndpoints:
    @pytest.fixture
    def app(self, mock_config):
        app = Flask(__name__)
        app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_list_feedback_requires_policy_id(self, client, mock_config):
        with patch.object(compliance_routes, "_config", mock_config):
            with patch.object(compliance_routes, "_critic_service", MagicMock()):
                resp = client.get("/api/v4/compliance/adaptive-feedback")
        assert resp.status_code == 400
        assert "policy_document_id" in resp.get_json()["error"]

    def test_list_feedback_returns_data(self, client, mock_config):
        mock_critic = MagicMock()
        mock_critic.get_feedback_history.return_value = [
            {"_id": "fb-1", "summary_evaluation": "OK", "suggestions": []}
        ]
        with patch.object(compliance_routes, "_config", mock_config):
            with patch.object(compliance_routes, "_critic_service", mock_critic):
                resp = client.get(
                    "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1"
                )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["feedback"]) == 1
        assert data["feedback"][0]["_id"] == "fb-1"

    def test_list_feedback_active_only(self, client, mock_config):
        mock_critic = MagicMock()
        mock_critic.get_feedback_history.return_value = []
        with patch.object(compliance_routes, "_config", mock_config):
            with patch.object(compliance_routes, "_critic_service", mock_critic):
                resp = client.get(
                    "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1&active_only=true"
                )
        assert resp.status_code == 200
        mock_critic.get_feedback_history.assert_called_once_with(
            policy_document_id="pol-1",
            active_only=True,
            limit=50,
            offset=0,
        )

    def test_list_feedback_log(self, client, mock_config):
        mock_critic = MagicMock()
        mock_critic.get_feedback_log.return_value = [
            {"_id": "log-1", "run_id": "run-1", "feedback_ids_used": ["fb-1"]}
        ]
        with patch.object(compliance_routes, "_config", mock_config):
            with patch.object(compliance_routes, "_critic_service", mock_critic):
                resp = client.get(
                    "/api/v4/compliance/adaptive-feedback/log?run_id=run-1"
                )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["log"]) == 1
        mock_critic.get_feedback_log.assert_called_once_with(
            run_id="run-1", limit=50, offset=0,
        )


# ---------------------------------------------------------------------------
# CriticService.ensure_indexes
# ---------------------------------------------------------------------------

class TestEnsureIndexes:
    def test_creates_indexes(self, critic, mock_mongo):
        coll_mock = MagicMock()
        mock_mongo.__getitem__.return_value.__getitem__.return_value = coll_mock
        critic.ensure_indexes()
        assert coll_mock.create_index.call_count >= 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _sync_run(func, *args, **kwargs):
    """Replacement for run_in_thread that runs synchronously."""
    return func(*args, **kwargs)


def _run_async(coro):
    """Run a coroutine with an explicit loop for Python 3.12 compatibility."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
