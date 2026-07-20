"""Regression tests for compliance drift detection orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from compliance_config import load_config
from compliance_suite_schemas import (
    DriftCheckRequest,
    GapAnalysisResponse,
    GapItem,
    HealthScoreResponse,
)
from compliance_suite_service import ComplianceSuiteService


def _gap_response(*gaps: GapItem) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="policy-1",
        company_name="Example Co",
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-07-20T10:00:00+00:00",
        gaps=list(gaps),
    )


def _build_service(
    *,
    current_gap: GapAnalysisResponse,
    previous_result: dict | None,
    new_chunks: list[dict],
    previous_check_at: datetime,
) -> tuple[ComplianceSuiteService, MagicMock, MagicMock, MagicMock]:
    config = load_config()
    config.compliance_database = "compliance"
    config.statute_database = "statutes"
    config.statute_sub_topic_embeddings_collection = "statute_chunks"
    config.statute_index_version_field = "indexed_at"
    config.statute_jurisdiction_field = "jurisdiction"
    config.default_jurisdictions = ["CA", "VA"]

    statute_collection = MagicMock()
    statute_collection.find.return_value = new_chunks
    statute_database = MagicMock()
    statute_database.__getitem__.return_value = statute_collection
    mongo_client = MagicMock()
    mongo_client.__getitem__.return_value = statute_database

    storage = MagicMock()
    storage.get_last_drift_check_at = AsyncMock(return_value=previous_check_at)
    storage.list_policy_document_ids = AsyncMock(return_value=["policy-1"])
    storage.get_last_compliance_result = AsyncMock(return_value=previous_result)
    storage.write_compliance_alert = AsyncMock()
    storage.set_last_drift_check_at = AsyncMock()

    gap_service = MagicMock()
    gap_service.run = AsyncMock(return_value=current_gap)
    rate_limiter = MagicMock()
    rate_limiter.allow = AsyncMock(return_value=True)

    service = ComplianceSuiteService(
        mongo_client=mongo_client,
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=storage,
        rate_limiter=rate_limiter,
        gap_analysis_v4_service=gap_service,
    )
    service.health_score = AsyncMock(
        return_value=HealthScoreResponse(
            policy_document_id="policy-1",
            privacy_health_score=70,
            analyzed_at="2026-07-20T10:00:00+00:00",
        )
    )
    return service, storage, gap_service, statute_collection


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_drift_check_detects_and_persists_material_changes(
    monkeypatch: pytest.MonkeyPatch,
    anyio_backend: str,
) -> None:
    """New missing gaps and score declines produce one persisted drift alert."""
    current_gap = _gap_response(
        GapItem(
            jurisdiction="CA",
            statute_reference="CA 100",
            requirement_summary="Access right",
            status="addressed",
        ),
        GapItem(
            jurisdiction="CA",
            statute_reference="CA 200",
            requirement_summary="Opt-out right",
            status="missing",
        ),
        GapItem(
            jurisdiction="CA",
            statute_reference="CA 300",
            requirement_summary="Notice right",
            status="addressed",
        ),
    )
    previous_result = {
        "privacy_health_score": 90,
        "gaps": [
            {
                "jurisdiction": "CA",
                "statute_reference": "CA 100",
                "requirement_summary": "Access right",
            },
            {
                "jurisdiction": "CA",
                "statute_reference": "CA 400",
                "requirement_summary": "Deletion right",
            },
        ],
    }
    last_check = datetime(2026, 7, 19, 9, 30, tzinfo=timezone.utc)
    service, storage, gap_service, statute_collection = _build_service(
        current_gap=current_gap,
        previous_result=previous_result,
        new_chunks=[
            {"_id": "chunk-1", "jurisdiction": "CA"},
            {"_id": "chunk-2", "jurisdiction": "CA"},
        ],
        previous_check_at=last_check,
    )
    monkeypatch.setattr("compliance_suite_service.uuid.uuid4", lambda: "alert-1")
    monkeypatch.setattr(
        "compliance_suite_service._iso",
        lambda dt=None: "2026-07-20T10:00:00+00:00",
    )

    result = await service.drift_check(
        DriftCheckRequest(
            since="2026-07-19T12:00:00Z",
            policy_document_ids=["policy-1"],
        )
    )

    statute_collection.find.assert_called_once_with(
        {"indexed_at": {"$gt": "2026-07-19T12:00:00+00:00"}},
        {"jurisdiction": 1, "_id": 1},
    )
    gap_request = gap_service.run.await_args.args[0]
    assert gap_request.policy_document_id == "policy-1"
    assert gap_request.applicable_jurisdictions == ["CA"]

    assert result.policies_checked == 1
    assert result.alerts_written == 1
    alert = result.alerts[0]
    assert alert.alert_id == "alert-1"
    assert [gap.requirement_summary for gap in alert.new_gaps] == ["Opt-out right"]
    assert [gap.requirement_summary for gap in alert.resolved_gaps] == ["Deletion right"]
    assert alert.affected_jurisdictions == ["CA"]
    assert alert.previous_score == 90
    assert alert.current_score == 70
    assert alert.score_delta == -20

    persisted = storage.write_compliance_alert.await_args.args[0]
    assert persisted["alert_id"] == "alert-1"
    assert persisted["new_gaps"] == [
        {
            "jurisdiction": "CA",
            "requirement_summary": "Opt-out right",
            "statute_reference": "CA 200",
        }
    ]
    assert persisted["resolved_gaps"] == [
        {
            "jurisdiction": "CA",
            "requirement_summary": "Deletion right",
            "statute_reference": "CA 400",
        }
    ]
    assert persisted["score_delta"] == -20
    storage.set_last_drift_check_at.assert_awaited_once_with()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_full_rebaseline_uses_all_policies_and_default_jurisdictions(
    anyio_backend: str,
) -> None:
    """A full rebaseline bypasses incremental lookup and still advances the checkpoint."""
    current_gap = _gap_response(
        GapItem(
            jurisdiction="CA",
            statute_reference="CA 100",
            requirement_summary="Access right",
            status="addressed",
        )
    )
    service, storage, gap_service, statute_collection = _build_service(
        current_gap=current_gap,
        previous_result=None,
        new_chunks=[{"_id": "should-not-be-read", "jurisdiction": "CO"}],
        previous_check_at=datetime(2026, 7, 19, tzinfo=timezone.utc),
    )

    result = await service.drift_check(DriftCheckRequest(full_rebaseline=True))

    statute_collection.find.assert_not_called()
    storage.list_policy_document_ids.assert_awaited_once_with("compliance")
    gap_request = gap_service.run.await_args.args[0]
    assert gap_request.applicable_jurisdictions == ["CA", "VA"]
    assert result.policies_checked == 1
    assert result.alerts == []
    storage.write_compliance_alert.assert_not_awaited()
    service.health_score.assert_not_awaited()
    storage.set_last_drift_check_at.assert_awaited_once_with()
