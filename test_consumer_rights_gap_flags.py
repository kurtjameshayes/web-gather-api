"""Regression tests for consumer-rights gap-flag coercion and per-type failures."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from compliance_config import load_config
from compliance_suite_schemas import ConsumerRightsRouterRequest
from compliance_suite_service import ComplianceSuiteService


def _service(llm: MagicMock) -> tuple[ComplianceSuiteService, MagicMock]:
    config = load_config()
    config.default_jurisdictions = ["CA"]
    config.llm_concurrency = 2
    storage = MagicMock()
    storage.write_compliance_result = AsyncMock()
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    service = ComplianceSuiteService(
        mongo_client=MagicMock(),
        config=config,
        retriever=MagicMock(),
        llm_client=llm,
        storage=storage,
        rate_limiter=limiter,
    )
    service._fetch_v4_statute_docs = AsyncMock(return_value={"CA": []})
    service._load_policy_from_chunks = AsyncMock(return_value=("Stored policy text.", "Acme"))
    return service, storage


def test_string_false_is_covered_and_non_dict_gaps_are_dropped() -> None:
    """bool() treats any non-empty string as covered, including "false" and "0"."""
    llm = MagicMock()
    llm.consumer_rights_router = AsyncMock(
        return_value={
            "trees": {"ca": {"id": "q1"}},
            "policy_gaps": {
                "ca": {"covered": "false", "gap": "missing deletion"},
                "zero_str": {"covered": "0", "gap": "  kept  "},
                "blank": {"covered": "", "gap": ""},
                "numeric_zero": {"covered": 0, "gap": None},
                "explicit_null": {"covered": None, "gap": "note"},
                "missing_flag": {"gap": "only a note"},
                "skipped": "not-a-dict",
            },
        }
    )
    service, storage = _service(llm)
    response = asyncio.run(
        service.consumer_rights_router(
            ConsumerRightsRouterRequest(
                text="We describe access rights.",
                policy_document_id="pol-9",
                applicable_jurisdictions=["CA"],
                request_types=["access"],
                save_results=True,
            )
        )
    )

    gaps = response.policy_gaps["access"]
    assert gaps["ca"].covered is True
    assert gaps["ca"].gap == "missing deletion"
    assert gaps["zero_str"].covered is True
    assert gaps["zero_str"].gap == "  kept  "
    assert gaps["blank"].covered is False
    assert gaps["blank"].gap == ""
    assert gaps["numeric_zero"].covered is False
    assert gaps["numeric_zero"].gap is None
    assert gaps["explicit_null"].covered is False
    assert gaps["missing_flag"].covered is False
    assert "skipped" not in gaps
    assert response.trees["access"] == {"ca": {"id": "q1"}}
    assert response.policy_document_id is None
    assert response.company_name is None
    assert response.request_types["access"] == "Right to Access / Know"
    storage.write_compliance_result.assert_not_called()
    service._load_policy_from_chunks.assert_not_called()
    service._fetch_v4_statute_docs.assert_awaited_once()
    assert service._fetch_v4_statute_docs.await_args.args[0] == ["CA"]
    assert service._fetch_v4_statute_docs.await_args.kwargs["categories"] == ["consumer_rights"]
    assert llm.consumer_rights_router.await_args.kwargs["request_type"] == "access"
    assert llm.consumer_rights_router.await_args.kwargs["request_type_label"] == "Right to Access / Know"


def test_failed_model_call_raises_before_save() -> None:
    """A gathered model exception is caught, then logged with an undefined logger.

    The NameError aborts the router. Later request types are not returned, and
    save_results does not persist.
    """
    llm = MagicMock()

    async def _route(*, request_type: str, **_kwargs: object) -> dict:
        if request_type == "deletion":
            raise RuntimeError("critic timeout")
        return {
            "trees": {"ca": {"id": "leaf"}},
            "policy_gaps": {"ca": {"covered": True, "gap": None}},
        }

    llm.consumer_rights_router = AsyncMock(side_effect=_route)
    service, storage = _service(llm)

    with pytest.raises(NameError, match="logger"):
        asyncio.run(
            service.consumer_rights_router(
                ConsumerRightsRouterRequest(
                    policy_document_id="pol-1",
                    applicable_jurisdictions=["CA"],
                    request_types=["deletion", "access"],
                    save_results=True,
                )
            )
        )

    assert llm.consumer_rights_router.await_count == 2
    storage.write_compliance_result.assert_not_called()


def test_none_model_result_raises_before_save() -> None:
    """A None model payload hits the same undefined logger and does not persist."""
    llm = MagicMock()
    llm.consumer_rights_router = AsyncMock(return_value=None)
    service, storage = _service(llm)

    with pytest.raises(NameError, match="logger"):
        asyncio.run(
            service.consumer_rights_router(
                ConsumerRightsRouterRequest(
                    policy_document_id="pol-1",
                    applicable_jurisdictions=["CA"],
                    request_types=["access"],
                    save_results=True,
                )
            )
        )

    storage.write_compliance_result.assert_not_called()
    service._load_policy_from_chunks.assert_awaited_once()


def test_numeric_gap_aborts_before_persistence() -> None:
    llm = MagicMock()
    llm.consumer_rights_router = AsyncMock(
        return_value={
            "trees": {"ca": {"id": "q1"}},
            "policy_gaps": {"ca": {"covered": False, "gap": 12}},
        }
    )
    service, storage = _service(llm)

    with pytest.raises(ValidationError):
        asyncio.run(
            service.consumer_rights_router(
                ConsumerRightsRouterRequest(
                    policy_document_id="pol-1",
                    applicable_jurisdictions=["CA"],
                    request_types=["access"],
                    save_results=True,
                )
            )
        )

    storage.write_compliance_result.assert_not_called()
    service._load_policy_from_chunks.assert_awaited_once()
