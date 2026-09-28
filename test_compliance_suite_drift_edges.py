"""Regression tests for drift-check alert rules that are easy to regress.

Pins timestamp fallback, which gap changes count as alerts, the 50-policy cap,
and the split between response jurisdictions and the stored alert.
"""
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
from compliance_suite_service import ComplianceSuiteService, ComplianceSuiteServiceError


def _gap_response(*gaps: GapItem, company_name: str = "Example Co") -> GapAnalysisResponse:
    return GapAnalysisResponse(
        policy_document_id="policy-1",
        company_name=company_name,
        applicable_jurisdictions=["CA"],
        analyzed_at="2026-07-20T10:00:00+00:00",
        gaps=list(gaps),
    )


def _gap(summary: str, status: str, jurisdiction: str = "CA", statute: str = "CA 100") -> GapItem:
    return GapItem(
        jurisdiction=jurisdiction,
        statute_reference=statute,
        requirement_summary=summary,
        status=status,
    )


def _health(score: int | None) -> HealthScoreResponse:
    return HealthScoreResponse(
        policy_document_id="policy-1",
        privacy_health_score=score,
        analyzed_at="2026-07-20T10:00:00+00:00",
    )


def _build_service(
    *,
    current_gap: GapAnalysisResponse,
    previous_result: dict | None,
    new_chunks: list[dict] | None = None,
    previous_check_at: datetime | str | None = None,
    use_v4: bool = True,
) -> tuple[ComplianceSuiteService, MagicMock, MagicMock, MagicMock]:
    config = load_config()
    config.compliance_database = "compliance"
    config.statute_database = "statutes"
    config.statute_sub_topic_embeddings_collection = "statute_chunks"
    config.statute_index_version_field = "indexed_at"
    config.statute_jurisdiction_field = "jurisdiction"
    config.default_jurisdictions = ["CA", "VA"]

    statute_collection = MagicMock()
    statute_collection.find.return_value = new_chunks or []
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
        gap_analysis_v4_service=gap_service if use_v4 else None,
    )
    service.health_score = AsyncMock(return_value=_health(70))
    if not use_v4:
        service.gap_analysis = AsyncMock(return_value=current_gap)
    return service, storage, gap_service, statute_collection


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_invalid_since_uses_stored_checkpoint_and_dedupes_jurisdictions(
    anyio_backend: str,
) -> None:
    """A bad since value is ignored. Chunk jurisdictions are first-seen and skip blanks."""
    checkpoint = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
    service, storage, gap_service, statute_collection = _build_service(
        current_gap=_gap_response(_gap("Access right", "addressed")),
        previous_result=None,
        new_chunks=[
            {"_id": "1", "jurisdiction": "CA"},
            {"_id": "2", "jurisdiction": "CA"},
            {"_id": "3", "jurisdiction": ""},
            {"_id": "4", "jurisdiction": None},
            {"_id": "5", "jurisdiction": "VA"},
        ],
        previous_check_at=checkpoint,
    )

    result = await service.drift_check(
        DriftCheckRequest(since="not-a-timestamp", policy_document_ids=["policy-1"])
    )

    statute_collection.find.assert_called_once_with(
        {"indexed_at": {"$gt": checkpoint.isoformat()}},
        {"jurisdiction": 1, "_id": 1},
    )
    gap_request = gap_service.run.await_args.args[0]
    assert gap_request.applicable_jurisdictions == ["CA", "VA"]
    assert result.alerts == []
    service.health_score.assert_not_awaited()
    storage.set_last_drift_check_at.assert_awaited_once_with()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_string_checkpoint_is_compared_as_text(anyio_backend: str) -> None:
    """A stored checkpoint without isoformat() is stringified into the $gt filter."""
    service, _, _, statute_collection = _build_service(
        current_gap=_gap_response(),
        previous_result=None,
        previous_check_at="2026-06-01T00:00:00",
    )

    await service.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    statute_collection.find.assert_called_once_with(
        {"indexed_at": {"$gt": "2026-06-01T00:00:00"}},
        {"jurisdiction": 1, "_id": 1},
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_score_drop_alerts_without_new_requirement_keys(anyio_backend: str) -> None:
    """A lower health score alerts even when every requirement key already existed."""
    service, storage, _, _ = _build_service(
        current_gap=_gap_response(_gap("Access right", "missing", statute="CA 100")),
        previous_result={
            "privacy_health_score": 80,
            "gaps": [
                {
                    "jurisdiction": "CA",
                    "statute_reference": "CA 100",
                    "requirement_summary": "Access right",
                    "status": "addressed",
                }
            ],
        },
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    service.health_score = AsyncMock(return_value=_health(60))

    result = await service.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    assert result.alerts_written == 1
    alert = result.alerts[0]
    assert alert.new_gaps == []
    assert alert.resolved_gaps == []
    assert alert.previous_score == 80
    assert alert.current_score == 60
    assert alert.score_delta == -20
    assert alert.trigger == "new_statute_indexed"
    persisted = storage.write_compliance_alert.await_args.args[0]
    assert persisted["new_gaps"] == []
    assert persisted["score_delta"] == -20


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_conflict_alerts_when_score_improves_but_partial_does_not(
    anyio_backend: str,
) -> None:
    """Only missing and conflict statuses are new gaps. A partial change is silent."""
    conflict_service, conflict_storage, _, _ = _build_service(
        current_gap=_gap_response(_gap("Sale opt-out", "conflict", statute="CA 200")),
        previous_result={"privacy_health_score": 40, "gaps": []},
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    conflict_service.health_score = AsyncMock(return_value=_health(70))

    conflict_result = await conflict_service.drift_check(
        DriftCheckRequest(policy_document_ids=["policy-1"])
    )

    assert conflict_result.alerts_written == 1
    assert [gap.requirement_summary for gap in conflict_result.alerts[0].new_gaps] == ["Sale opt-out"]
    assert conflict_result.alerts[0].score_delta == 30
    conflict_storage.write_compliance_alert.assert_awaited()

    partial_service, partial_storage, _, _ = _build_service(
        current_gap=_gap_response(_gap("Sale opt-out", "partial", statute="CA 200")),
        previous_result={"privacy_health_score": 70, "gaps": []},
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    partial_service.health_score = AsyncMock(return_value=_health(70))

    partial_result = await partial_service.drift_check(
        DriftCheckRequest(policy_document_ids=["policy-1"])
    )

    assert partial_result.alerts == []
    partial_storage.write_compliance_alert.assert_not_awaited()
    partial_service.health_score.assert_awaited()
    partial_storage.set_last_drift_check_at.assert_awaited_once_with()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_resolved_only_change_does_not_alert_until_score_drops(anyio_backend: str) -> None:
    """Disappeared requirements are resolved gaps, but they do not alert on their own."""
    stable, stable_storage, _, _ = _build_service(
        current_gap=_gap_response(),
        previous_result={
            "privacy_health_score": 70,
            "gaps": [
                {
                    "jurisdiction": "CA",
                    "statute_reference": "CA 400",
                    "requirement_summary": "Deletion right",
                }
            ],
        },
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    stable.health_score = AsyncMock(return_value=_health(70))

    stable_result = await stable.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    assert stable_result.alerts == []
    stable_storage.write_compliance_alert.assert_not_awaited()

    dropped, dropped_storage, _, _ = _build_service(
        current_gap=_gap_response(),
        previous_result={
            "privacy_health_score": 80,
            "gaps": [
                {
                    "jurisdiction": "CA",
                    "statute_reference": "CA 400",
                    "requirement_summary": "Deletion right",
                }
            ],
        },
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    dropped.health_score = AsyncMock(return_value=_health(50))

    dropped_result = await dropped.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    assert dropped_result.alerts_written == 1
    alert = dropped_result.alerts[0]
    assert alert.new_gaps == []
    assert [gap.requirement_summary for gap in alert.resolved_gaps] == ["Deletion right"]
    assert alert.score_delta == -30
    persisted = dropped_storage.write_compliance_alert.await_args.args[0]
    assert persisted["resolved_gaps"][0]["statute_reference"] == "CA 400"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_empty_statute_scan_splits_response_and_stored_jurisdictions(
    anyio_backend: str,
) -> None:
    """No new chunks uses default jurisdictions for analysis.

    The response alert fills jurisdictions from current gaps. The stored alert keeps
    the empty statute list.
    """
    service, storage, gap_service, statute_collection = _build_service(
        current_gap=_gap_response(_gap("Notice", "missing", jurisdiction="CO", statute="CO 1")),
        previous_result={"privacy_health_score": 50, "gaps": []},
        new_chunks=[],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    service.health_score = AsyncMock(return_value=_health(50))

    result = await service.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    statute_collection.find.assert_called_once()
    gap_request = gap_service.run.await_args.args[0]
    assert gap_request.applicable_jurisdictions == ["CA", "VA"]
    assert result.alerts_written == 1
    assert result.alerts[0].affected_jurisdictions == ["CO"]
    assert result.alerts[0].trigger == "new_statute_indexed"
    persisted = storage.write_compliance_alert.await_args.args[0]
    assert persisted["affected_jurisdictions"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_gap_failure_skips_policy_and_health_failure_still_alerts(
    anyio_backend: str,
) -> None:
    """A gap-analysis error skips that policy. A health-score error still records new gaps."""
    current_gap = _gap_response(_gap("Access right", "missing"))
    service, storage, gap_service, _ = _build_service(
        current_gap=current_gap,
        previous_result={"privacy_health_score": 90, "gaps": []},
        new_chunks=[{"_id": "1", "jurisdiction": "CA"}],
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    gap_service.run = AsyncMock(side_effect=[RuntimeError("llm down"), current_gap])
    service.health_score = AsyncMock(side_effect=RuntimeError("score down"))

    result = await service.drift_check(
        DriftCheckRequest(policy_document_ids=["policy-a", "policy-b"])
    )

    assert result.policies_checked == 2
    assert result.alerts_written == 1
    alert = result.alerts[0]
    assert alert.policy_document_id == "policy-b"
    assert [gap.requirement_summary for gap in alert.new_gaps] == ["Access right"]
    assert alert.previous_score == 90
    assert alert.current_score is None
    assert alert.score_delta is None
    assert storage.get_last_compliance_result.await_count == 2
    service.health_score.assert_awaited_once()
    storage.set_last_drift_check_at.assert_awaited_once_with()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_policy_list_is_capped_and_empty_list_loads_stored_ids(anyio_backend: str) -> None:
    """Only the first 50 explicit policy ids are checked. [] loads ids from storage."""
    service, storage, gap_service, _ = _build_service(
        current_gap=_gap_response(),
        previous_result=None,
        previous_check_at=None,
    )
    policy_ids = [f"policy-{index}" for index in range(51)]

    capped = await service.drift_check(DriftCheckRequest(policy_document_ids=policy_ids))

    assert capped.policies_checked == 50
    assert gap_service.run.await_count == 50
    storage.list_policy_document_ids.assert_not_awaited()

    storage.list_policy_document_ids.reset_mock()
    gap_service.run.reset_mock()
    empty = await service.drift_check(
        DriftCheckRequest(policy_document_ids=[], full_rebaseline=True)
    )

    storage.list_policy_document_ids.assert_awaited_once_with("compliance")
    assert empty.policies_checked == 1
    gap_service.run.assert_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_missing_v4_service_uses_gap_analysis(anyio_backend: str) -> None:
    """Without a v4 service, drift re-runs the suite gap analysis."""
    service, _, gap_service, _ = _build_service(
        current_gap=_gap_response(_gap("Access right", "addressed")),
        previous_result=None,
        previous_check_at=None,
        use_v4=False,
    )

    await service.drift_check(
        DriftCheckRequest(full_rebaseline=True, policy_document_ids=["policy-1"])
    )

    service.gap_analysis.assert_awaited_once()
    request = service.gap_analysis.await_args.args[0]
    assert request.policy_document_id == "policy-1"
    assert request.applicable_jurisdictions == ["CA", "VA"]
    gap_service.run.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_rate_limit_stops_before_storage(anyio_backend: str) -> None:
    """A denied rate limit is 429 and does not read checkpoints or policies."""
    service, storage, gap_service, statute_collection = _build_service(
        current_gap=_gap_response(),
        previous_result={"gaps": []},
        previous_check_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    service._rate_limiter.allow = AsyncMock(return_value=False)

    with pytest.raises(ComplianceSuiteServiceError) as exc_info:
        await service.drift_check(DriftCheckRequest(policy_document_ids=["policy-1"]))

    assert exc_info.value.status_code == 429
    assert "Rate limit" in str(exc_info.value)
    storage.get_last_drift_check_at.assert_not_awaited()
    statute_collection.find.assert_not_called()
    gap_service.run.assert_not_called()
    storage.set_last_drift_check_at.assert_not_awaited()
