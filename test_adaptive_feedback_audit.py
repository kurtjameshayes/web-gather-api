"""Audit-trail and access checks for adaptive feedback history.

Covers critic suggestion sanitization, the supersede write, history and
consumption-log queries, and the v4 list endpoints' auth and page bounds.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

import compliance_routes
from adaptive_feedback_service import CriticService
from compliance_config import load_config
from compliance_routes_v4 import compliance_v4_bp
from compliance_suite_schemas import GapAnalysisResponse, GapItem, GapSummary


def _critic() -> tuple[CriticService, MagicMock]:
    config = load_config()
    config.adaptive_feedback_enabled = True
    config.adaptive_feedback_max_items = 5
    config.adaptive_feedback_collection = "adaptive_feedback"
    config.adaptive_feedback_log_collection = "adaptive_feedback_log"
    config.adaptive_critic_prompt_path = "prompts/adaptive_critic.yaml"
    config.compliance_database = "privacy-compliance"
    mongo = MagicMock()
    critic = CriticService(mongo_client=mongo, config=config, llm_client=MagicMock())
    return critic, mongo


def _cursor(docs: list) -> MagicMock:
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = docs
    return cursor


def _bind_collection(mongo: MagicMock, coll: MagicMock) -> None:
    mongo.__getitem__.return_value.__getitem__.return_value = coll


class _ObjectId:
    def __str__(self) -> str:
        return "507f1f77bcf86cd799439011"


def _sample_response() -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="pol-1",
        policy_document_ids=["pol-1", "pol-2"],
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
            )
        ],
        summary=GapSummary(total_requirements=1, addressed=1),
    )


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


class TestSuggestionSanitization:
    def test_invalid_items_consume_the_raw_cap(self) -> None:
        raw = [
            {"category": "not-a-category", "instruction": f"skip {i}"}
            for i in range(10)
        ]
        raw.append(
            {
                "category": "false_positive",
                "instruction": "This valid item is past the cap",
                "severity": "high",
            }
        )
        assert CriticService._validate_suggestions(raw) == []

    def test_keeps_neighbors_and_normalizes_fields(self) -> None:
        raw = [
            "not-a-dict",
            {
                "category": " Citation_Quality ",
                "severity": " HIGH ",
                "instruction": "  keep this  ",
                "description": "  trimmed  ",
                "statute_reference": "",
            },
            {
                "category": "false_negative",
                "instruction": "   ",
                "severity": "low",
            },
            {
                "category": "confidence_calibration",
                "instruction": "Calibrate low confidence",
                "statute_reference": "  § 1  ",
            },
        ]
        result = CriticService._validate_suggestions(raw)
        assert result == [
            {
                "category": "citation_quality",
                "description": "trimmed",
                "instruction": "keep this",
                "statute_reference": None,
                "severity": "high",
            },
            {
                "category": "confidence_calibration",
                "description": "",
                "instruction": "Calibrate low confidence",
                "statute_reference": "  § 1  ",
                "severity": "medium",
            },
        ]


class TestFeedbackQueries:
    def test_history_active_only_filters_and_stringifies_ids(self) -> None:
        critic, mongo = _critic()
        docs = [
            {"_id": _ObjectId(), "policy_document_id": "pol-1"},
            {"policy_document_id": "pol-1"},
        ]
        coll = MagicMock()
        coll.find.return_value = _cursor(docs)
        _bind_collection(mongo, coll)

        result = critic.get_feedback_history(
            "pol-1", active_only=True, limit=10, offset=4
        )

        assert result[0]["_id"] == "507f1f77bcf86cd799439011"
        assert "_id" not in result[1]
        coll.find.assert_called_once_with(
            {"policy_document_id": "pol-1", "superseded_by": None}
        )
        cursor = coll.find.return_value
        cursor.sort.assert_called_once_with("created_at", -1)
        cursor.skip.assert_called_once_with(4)
        cursor.limit.assert_called_once_with(10)

    def test_history_without_active_only_returns_superseded_rows(self) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        coll.find.return_value = _cursor([])
        _bind_collection(mongo, coll)

        critic.get_feedback_history("pol-9", active_only=False, limit=50, offset=0)

        coll.find.assert_called_once_with({"policy_document_id": "pol-9"})

    def test_log_filters_only_when_run_id_is_present(self) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        empty_cursor = _cursor([])
        match_cursor = _cursor([{"_id": 15, "run_id": "run-1"}])
        padded_cursor = _cursor([])
        coll.find.side_effect = [empty_cursor, match_cursor, padded_cursor]
        _bind_collection(mongo, coll)

        blank = critic.get_feedback_log(run_id="", limit=7, offset=2)
        found = critic.get_feedback_log(run_id="run-1", limit=7, offset=2)
        critic.get_feedback_log(run_id="  ", limit=7, offset=2)

        assert blank == []
        assert found[0]["_id"] == "15"
        assert coll.find.call_args_list[0].args == ({},)
        assert coll.find.call_args_list[1].args == ({"run_id": "run-1"},)
        assert coll.find.call_args_list[2].args == ({"run_id": "  "},)
        match_cursor.skip.assert_called_once_with(2)
        match_cursor.limit.assert_called_once_with(7)

    def test_active_feedback_limit_zero_uses_configured_max(self) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        coll.find.return_value = _cursor([])
        _bind_collection(mongo, coll)

        critic.get_active_feedback("pol-1", limit=0)
        coll.find.return_value.limit.assert_called_with(5)

        critic.get_active_feedback("pol-1", limit=3)
        coll.find.return_value.limit.assert_called_with(3)
        coll.find.assert_called_with(
            {"policy_document_id": "pol-1", "superseded_by": None}
        )


class TestEvaluateSupersede:
    def test_supersedes_only_the_primary_policy_even_when_suggestions_are_empty(
        self,
    ) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        _bind_collection(mongo, coll)
        critic._llm._call_json = AsyncMock(
            return_value={
                "summary_evaluation": "  needs work  ",
                "suggestions": [
                    {"category": "false_positive", "instruction": "   "},
                    "bad",
                ],
            }
        )

        with patch("adaptive_feedback_service.run_in_thread", side_effect=_sync_run):
            feedback_id = asyncio.run(critic.evaluate(_sample_response(), "run-9"))

        inserted = coll.insert_one.call_args[0][0]
        assert feedback_id == inserted["_id"]
        assert inserted["policy_document_id"] == "pol-1"
        assert inserted["policy_document_ids"] == ["pol-1", "pol-2"]
        assert inserted["suggestions"] == []
        assert inserted["summary_evaluation"] == "needs work"
        assert inserted["superseded_by"] is None
        coll.update_many.assert_called_once_with(
            {
                "policy_document_id": "pol-1",
                "superseded_by": None,
                "_id": {"$ne": feedback_id},
            },
            {"$set": {"superseded_by": feedback_id}},
        )

    def test_llm_exception_does_not_write_or_supersede(self) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        _bind_collection(mongo, coll)
        critic._llm._call_json = AsyncMock(side_effect=RuntimeError("critic down"))

        result = asyncio.run(critic.evaluate(_sample_response(), "run-9"))

        assert result is None
        coll.insert_one.assert_not_called()
        coll.update_many.assert_not_called()


class TestEnsureIndexes:
    def test_index_failure_does_not_raise(self) -> None:
        critic, mongo = _critic()
        coll = MagicMock()
        coll.create_index.side_effect = RuntimeError("index build failed")
        _bind_collection(mongo, coll)

        critic.ensure_indexes()


def _app(config):
    app = Flask(__name__)
    app.register_blueprint(compliance_v4_bp, url_prefix="/api/v4/compliance")
    return app


class TestAdaptiveFeedbackEndpointGuards:
    @pytest.fixture
    def secured_config(self):
        config = load_config()
        config.auth_required = True
        config.api_key = "secret"
        config.allowed_roles = ["admin"]
        config.default_role_header = "x-role"
        return config

    def test_history_rejects_missing_and_wrong_keys_before_query(
        self, secured_config
    ) -> None:
        critic = MagicMock()
        client = _app(secured_config).test_client()
        with patch.object(compliance_routes, "_config", secured_config):
            with patch.object(compliance_routes, "_critic_service", critic):
                missing = client.get(
                    "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1"
                )
                wrong = client.get(
                    "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1",
                    headers={"x-api-key": "nope", "x-role": "admin"},
                )
                blocked_role = client.get(
                    "/api/v4/compliance/adaptive-feedback?policy_document_id=pol-1",
                    headers={"x-api-key": "secret", "x-role": "viewer"},
                )
        assert missing.status_code == 401
        assert missing.get_json()["error"] == "Missing API key."
        assert wrong.status_code == 403
        assert wrong.get_json()["error"] == "Invalid API key."
        assert blocked_role.status_code == 403
        assert blocked_role.get_json()["error"] == "Insufficient role."
        critic.get_feedback_history.assert_not_called()

    def test_log_rejects_missing_key_before_query(self, secured_config) -> None:
        critic = MagicMock()
        client = _app(secured_config).test_client()
        with patch.object(compliance_routes, "_config", secured_config):
            with patch.object(compliance_routes, "_critic_service", critic):
                resp = client.get("/api/v4/compliance/adaptive-feedback/log")
        assert resp.status_code == 401
        critic.get_feedback_log.assert_not_called()

    def test_history_clamps_pages_and_parses_active_only(self, secured_config) -> None:
        critic = MagicMock()
        critic.get_feedback_history.return_value = []
        client = _app(secured_config).test_client()
        headers = {"x-api-key": "  secret  ", "x-role": "Admin"}
        with patch.object(compliance_routes, "_config", secured_config):
            with patch.object(compliance_routes, "_critic_service", critic):
                high = client.get(
                    "/api/v4/compliance/adaptive-feedback"
                    "?policy_document_id=pol-1&active_only=YES&limit=999&offset=-4",
                    headers=headers,
                )
                invalid = client.get(
                    "/api/v4/compliance/adaptive-feedback"
                    "?policy_document_id=pol-1&active_only=%20yes%20&limit=nope&offset=abc",
                    headers=headers,
                )
                zero = client.get(
                    "/api/v4/compliance/adaptive-feedback"
                    "?policy_document_id=pol-1&active_only=0&limit=0&offset=0",
                    headers=headers,
                )
        assert high.status_code == 200
        assert high.get_json() == {"feedback": [], "limit": 200, "offset": 0}
        assert invalid.get_json()["limit"] == 50
        assert invalid.get_json()["offset"] == 0
        assert zero.get_json() == {"feedback": [], "limit": 1, "offset": 0}
        assert critic.get_feedback_history.call_args_list[0].kwargs == {
            "policy_document_id": "pol-1",
            "active_only": True,
            "limit": 200,
            "offset": 0,
        }
        assert critic.get_feedback_history.call_args_list[1].kwargs == {
            "policy_document_id": "pol-1",
            "active_only": False,
            "limit": 50,
            "offset": 0,
        }
        assert critic.get_feedback_history.call_args_list[2].kwargs["active_only"] is False
        assert critic.get_feedback_history.call_args_list[2].kwargs["limit"] == 1

    def test_log_clamps_pages_and_passes_run_id_through(self, secured_config) -> None:
        critic = MagicMock()
        critic.get_feedback_log.return_value = []
        client = _app(secured_config).test_client()
        headers = {"x-api-key": "secret", "x-role": "admin"}
        with patch.object(compliance_routes, "_config", secured_config):
            with patch.object(compliance_routes, "_critic_service", critic):
                bounded = client.get(
                    "/api/v4/compliance/adaptive-feedback/log?run_id=run-1&limit=0&offset=-1",
                    headers=headers,
                )
                blank = client.get(
                    "/api/v4/compliance/adaptive-feedback/log?run_id=&limit=10.5&offset=%20%20",
                    headers=headers,
                )
                missing = client.get(
                    "/api/v4/compliance/adaptive-feedback/log",
                    headers=headers,
                )
        assert bounded.status_code == 200
        assert bounded.get_json() == {"log": [], "limit": 1, "offset": 0}
        assert blank.get_json()["limit"] == 50
        assert blank.get_json()["offset"] == 0
        assert critic.get_feedback_log.call_args_list[0].kwargs == {
            "run_id": "run-1",
            "limit": 1,
            "offset": 0,
        }
        assert critic.get_feedback_log.call_args_list[1].kwargs == {
            "run_id": "",
            "limit": 50,
            "offset": 0,
        }
        assert critic.get_feedback_log.call_args_list[2].kwargs["run_id"] is None
