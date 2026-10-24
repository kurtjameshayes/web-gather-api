"""V4 runs whose mappings are all unusable still persist and invoke the critic.

An empty category_mapping collection returns before any write. A non-empty
collection of definitions rows and rows with no policy categories is different:
the service skips every statute lookup, then still stores an empty gap result
and evaluates the critic. That can supersede real feedback with an analysis
that never called the model.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from adaptive_feedback_service import CriticService
from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4


class _Coll:
    def __init__(self) -> None:
        self.find_one_calls: List[Dict[str, Any]] = []
        self.find_calls: List[tuple] = []
        self.count_calls: List[Dict[str, Any]] = []
        self.inserts: List[Dict[str, Any]] = []
        self.find_one_result: Optional[Dict[str, Any]] = None
        self.find_result: List[Dict[str, Any]] = []
        self.count_result = 0

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.find_one_calls.append(query)
        return self.find_one_result

    def find(self, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.find_calls.append((query, projection))
        return list(self.find_result)

    def count_documents(self, query: Dict[str, Any]) -> int:
        self.count_calls.append(query)
        return self.count_result

    def insert_one(self, doc: Dict[str, Any]) -> None:
        self.inserts.append(doc)


class _DB:
    def __init__(self) -> None:
        self.colls: Dict[str, _Coll] = {}

    def __getitem__(self, name: str) -> _Coll:
        return self.colls.setdefault(name, _Coll())


class _Mongo:
    def __init__(self) -> None:
        self.dbs: Dict[str, _DB] = {}

    def __getitem__(self, name: str) -> _DB:
        return self.dbs.setdefault(name, _DB())


def test_unusable_mappings_persist_empty_run_and_still_evaluate_critic() -> None:
    cfg = load_config()
    cfg.adaptive_feedback_enabled = True
    mongo = _Mongo()
    db = mongo[cfg.compliance_database]
    policies = db[cfg.policies_collection]
    statutes = db[cfg.statute_sub_topic_embeddings_collection]
    legal = db[cfg.policy_legal_embeddings_collection]
    mappings = db[cfg.category_mapping_collection]
    results = db[cfg.compliance_results_collection]
    logs = db[cfg.compliance_run_log_collection]

    policies.find_one_result = {
        cfg.policy_document_id_field: "pol-1",
        "text": "We keep data.",
        "company_name": "Acme",
    }
    legal.count_result = 1
    mappings.find_result = [
        {
            "statute_category": "definitions",
            "policy_categories": ["notice"],
            "sub_topic": "defined",
        },
        {
            "statute_category": "consumer_rights",
            "policy_categories": [],
            "sub_topic": "empty",
        },
        {
            "statute_category": "controller_duties",
            "sub_topic": "missing-cats",
        },
    ]
    statutes.find_result = [{"_id": "should-not-load", "subtopic_text": "Do not analyze"}]

    feedback_docs = [
        {
            "_id": "fb-1",
            "suggestions": [{"instruction": "Prefer deletion quotes"}],
        }
    ]
    critic = MagicMock()
    critic.get_active_feedback.return_value = feedback_docs
    critic.format_feedback_for_prompt.side_effect = lambda docs: CriticService.format_feedback_for_prompt(
        None, docs
    )
    critic.record_feedback_usage = AsyncMock()
    critic.evaluate = AsyncMock()

    llm = MagicMock()
    llm.gap_check_v4 = AsyncMock()
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=True)
    service = GapAnalysisServiceV4(mongo, cfg, llm, limiter, critic=critic)

    response = asyncio.run(
        service.run(
            GapAnalysisRequest(
                policy_document_id="pol-1",
                applicable_jurisdictions=["CA"],
                save_results=True,
            )
        )
    )

    assert response.gaps == []
    assert response.company_name == "Acme"
    assert response.statute_chunk_ids_used is None
    assert response.retrieval_metadata.statute_items_considered == 0
    assert response.retrieval_metadata.statute_pairs_matched == 0
    assert response.summary.total_requirements == 0
    assert statutes.find_calls == []
    assert legal.find_calls == []
    assert llm.gap_check_v4.await_count == 0
    assert legal.count_calls == [{cfg.policy_document_id_field: {"$in": ["pol-1"]}}]
    critic.get_active_feedback.assert_called_once_with("pol-1")

    assert len(results.inserts) == 1
    stored = results.inserts[0]
    assert stored["gaps"] == []
    assert stored["policy_document_id"] == "pol-1"
    assert stored["company_name"] == "Acme"
    assert stored["run_type"] == "gap_analysis_v4"
    assert stored["run_types"] == ["gap_v4"]
    assert "policy_document_ids" not in stored
    assert len(logs.inserts) == 1
    assert logs.inserts[0]["statute_item_ids"] == []
    assert logs.inserts[0]["policy_document_id"] == "pol-1"

    rendered = (
        "PRIOR ANALYSIS FEEDBACK (incorporate these lessons into your analysis):\n\n"
        "- Prefer deletion quotes"
    )
    critic.record_feedback_usage.assert_awaited_once_with(
        run_id=stored["_id"],
        feedback_ids=["fb-1"],
        rendered_text=rendered,
    )
    critic.evaluate.assert_awaited_once()
    evaluated, run_id = critic.evaluate.await_args.args
    assert run_id == stored["_id"]
    assert evaluated.gaps == []
    assert evaluated.policy_document_id == "pol-1"
