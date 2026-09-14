"""ComplianceSuiteService._fetch_v4_statute_docs bucketing.

GapAnalysisServiceV4 has a sibling helper covered elsewhere. Citations,
risk assessment, and the consumer-rights router use this suite copy.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from compliance_config import load_config
from compliance_suite_service import ComplianceSuiteService


def _service(mongo: MagicMock) -> ComplianceSuiteService:
    config = load_config()
    return ComplianceSuiteService(
        mongo_client=mongo,
        config=config,
        retriever=MagicMock(),
        llm_client=MagicMock(),
        storage=MagicMock(),
        rate_limiter=MagicMock(),
    )


def _wire_find(mongo: MagicMock, docs: list) -> MagicMock:
    cursor = MagicMock()
    cursor.limit.return_value = docs
    coll = MagicMock()
    coll.find.return_value = cursor
    mongo.__getitem__.return_value.__getitem__.return_value = coll
    return coll


def test_suite_fetch_v4_defaults_analyze_categories_and_limit() -> None:
    mongo = MagicMock()
    coll = _wire_find(mongo, [])
    svc = _service(mongo)

    result = asyncio.run(svc._fetch_v4_statute_docs(["CA"]))

    assert result == {"CA": []}
    coll.find.assert_called_once_with(
        {"category": {"$in": ["consumer_rights", "controller_duties"]}}
    )
    coll.find.return_value.limit.assert_called_once_with(200)


def test_suite_fetch_v4_broadcasts_blank_jurisdiction() -> None:
    mongo = MagicMock()
    blank = {"document_id": "stat-blank", "jurisdiction": "  ", "requirement_summary": "x"}
    california = {"document_id": "stat-ca", "jurisdiction": "California", "requirement_summary": "y"}
    _wire_find(mongo, [blank, california])
    svc = _service(mongo)

    result = asyncio.run(svc._fetch_v4_statute_docs(["CA", "VA"]))

    assert blank in result["CA"]
    assert blank in result["VA"]
    assert california in result["CA"]
    assert california not in result["VA"]


def test_suite_fetch_v4_custom_categories_replace_defaults() -> None:
    mongo = MagicMock()
    coll = _wire_find(mongo, [])
    svc = _service(mongo)

    asyncio.run(
        svc._fetch_v4_statute_docs(["CA"], categories=["consumer_rights"], limit=10)
    )

    coll.find.assert_called_once_with({"category": {"$in": ["consumer_rights"]}})
    coll.find.return_value.limit.assert_called_once_with(10)
