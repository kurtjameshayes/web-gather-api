"""Regression tests for v4 mapping fallback, citation binding, and policy filters.

These contracts sit beside the corpus/prompt coverage in the v4 service and are
not asserted there: a wrong fallback silently returns no gaps, and citation
binding decides whether an addressed quote survives.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

import pytest
from bson import ObjectId
from pydantic import ValidationError

from compliance_config import load_config
from compliance_suite_schemas import GapAnalysisRequest
from gap_analysis_service_v4 import GapAnalysisServiceV4, GapAnalysisServiceV4Error


class _Coll:
    def __init__(self) -> None:
        self.find_ones: List[Dict[str, Any]] = []
        self.finds: List[tuple] = []
        self.counts: List[Dict[str, Any]] = []
        self.inserts: List[Dict[str, Any]] = []
        self.find_one_impl: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]] = lambda _q: None
        self.find_impl: Callable[..., List[Dict[str, Any]]] = lambda _q, _proj=None: []
        self.count_impl: Callable[[Dict[str, Any]], int] = lambda _q: 0

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.find_ones.append(query)
        return self.find_one_impl(query)

    def find(self, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.finds.append((query, projection))
        return self.find_impl(query, projection)

    def count_documents(self, query: Dict[str, Any]) -> int:
        self.counts.append(query)
        return self.count_impl(query)

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


def _run(coro):
    return asyncio.run(coro)


def _config(**overrides):
    cfg = load_config()
    cfg.adaptive_feedback_enabled = False
    cfg.compliance_database = "privacy-compliance"
    cfg.statute_database = ""
    cfg.category_mapping_collection = "category_mapping"
    cfg.default_jurisdictions = ["CA"]
    cfg.policies_collection = "policies"
    cfg.policy_legal_embeddings_collection = "policy_legal_embeddings"
    cfg.statute_sub_topic_embeddings_collection = "statute_sub_topic_embeddings"
    cfg.compliance_results_collection = "compliance_results"
    cfg.compliance_run_log_collection = "compliance_run_log"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _service(mongo, llm_result: Any = None, *, allow: bool = True, **cfg_overrides):
    llm = MagicMock()
    if isinstance(llm_result, list):
        llm.gap_check_v4 = AsyncMock(side_effect=llm_result)
    else:
        llm.gap_check_v4 = AsyncMock(return_value=llm_result or {})
    limiter = MagicMock()
    limiter.allow = AsyncMock(return_value=allow)
    service = GapAnalysisServiceV4(mongo, _config(**cfg_overrides), llm, limiter, critic=None)
    return service, llm


def _world() -> tuple[_Mongo, Dict[str, _Coll]]:
    mongo = _Mongo()
    db = mongo["privacy-compliance"]
    cols = {
        "policies": db["policies"],
        "legal": db["policy_legal_embeddings"],
        "mapping": db["category_mapping"],
        "plural": db["category_mappings"],
        "statutes": db["statute_sub_topic_embeddings"],
        "results": db["compliance_results"],
        "logs": db["compliance_run_log"],
    }
    return mongo, cols


def _policy(pid: str = "pol-1", text: str = "We honor deletion requests.", company: str = "Acme", **extra):
    doc = {"document_id": pid, "text": text, "company_name": company}
    doc.update(extra)
    return doc


def _mapping(
    category: str = "consumer_rights",
    policy_categories: Any = ("rights",),
    sub_topic: str = "delete",
):
    categories = ["rights"] if policy_categories == ("rights",) else policy_categories
    return {
        "statute_category": category,
        "policy_categories": categories,
        "sub_topic": sub_topic,
    }


def _statute(_id: str = "s1", **extra) -> Dict[str, Any]:
    doc = {
        "_id": _id,
        "header_text": "Civ. Code 1798.105",
        "subtopic_text": "A consumer may request deletion.",
        "requirement_summary": "Right to delete",
        "jurisdiction": "CA",
    }
    doc.update(extra)
    return doc


def _llm(**overrides) -> Dict[str, Any]:
    payload = {
        "status": "addressed",
        "policy_quote": "We honor deletion requests.",
        "confidence": "high",
        "requirement_summary": "Right to delete",
        "statute_quote": "Consumers may delete.",
        "conflict_description": "Aligned.",
        "_analysis_failed": False,
    }
    payload.update(overrides)
    return payload


def _seed_one(cols: Dict[str, _Coll], policy: Optional[Dict[str, Any]] = None, statutes: Optional[List[Dict[str, Any]]] = None) -> None:
    stored = policy or _policy()
    pid = stored["document_id"]

    def find_one(query: Dict[str, Any], _doc=stored, _pid=pid):
        if query.get("document_id") == _pid:
            return _doc
        return None

    cols["policies"].find_one_impl = find_one
    cols["legal"].count_impl = lambda _q: 1
    cols["mapping"].find_impl = lambda _q, _proj=None: [_mapping()]
    cols["statutes"].find_impl = lambda _q, _proj=None: statutes or [_statute()]
    cols["legal"].find_impl = lambda _q, _proj=None: [{"chunk_text": "We honor deletion requests."}]


def _req(**overrides) -> GapAnalysisRequest:
    payload = {"policy_document_id": "pol-1", "applicable_jurisdictions": ["CA"], "save_results": True}
    payload.update(overrides)
    return GapAnalysisRequest.model_validate(payload)


def test_plural_mapping_collection_is_used_only_for_the_default_name() -> None:
    mongo, cols = _world()
    _seed_one(cols)
    cols["mapping"].find_impl = lambda _q, _proj=None: []
    cols["plural"].find_impl = lambda _q, _proj=None: [_mapping()]
    service, llm = _service(mongo, _llm())

    response = _run(service.run(_req()))

    assert llm.gap_check_v4.await_count == 1
    assert cols["plural"].finds
    assert response.gaps[0].status == "addressed"
    assert response.summary.addressed == 1

    custom, custom_cols = _world()
    _seed_one(custom_cols)
    custom_cols["mapping"].find_impl = lambda _q, _proj=None: [_mapping()]
    custom_db = custom["privacy-compliance"]
    custom_db["custom_map"].find_impl = lambda _q, _proj=None: []
    custom_cols["plural"].find_impl = lambda _q, _proj=None: [_mapping()]
    custom_service, custom_llm = _service(custom, _llm(), category_mapping_collection="custom_map")

    empty = _run(custom_service.run(_req()))

    assert custom_llm.gap_check_v4.await_count == 0
    assert custom_cols["plural"].finds == []
    assert empty.gaps == []
    assert custom_cols["results"].inserts == []
    assert custom_cols["statutes"].finds == []


def test_empty_mappings_return_company_and_requested_jurisdictions_without_writes() -> None:
    mongo, cols = _world()
    _seed_one(cols)
    cols["mapping"].find_impl = lambda _q, _proj=None: []
    service, llm = _service(mongo, _llm())

    response = _run(service.run(_req(applicable_jurisdictions=["VA"])))

    assert llm.gap_check_v4.await_count == 0
    assert response.gaps == []
    assert response.company_name == "Acme"
    assert response.applicable_jurisdictions == ["VA"]
    assert response.statute_chunk_ids_used is None
    assert response.retrieval_metadata.statute_items_considered == 0
    assert response.run_type == "gap_analysis_v4"
    assert cols["results"].inserts == []
    assert cols["logs"].inserts == []
    assert cols["statutes"].finds == []


def test_citation_binding_demotes_only_addressed_and_prefers_full_policy_text() -> None:
    mongo, cols = _world()
    statutes = [
        _statute("exact"),
        _statute("collapsed"),
        _statute("unbound-addressed"),
        _statute("conflict"),
        _statute("partial"),
        _statute("ambiguous"),
        _statute("missing"),
        _statute("failed"),
        _statute("blank-quote"),
        _statute("whitespace-quote"),
        _statute("chunk-only"),
    ]
    _seed_one(cols, statutes=statutes)
    cols["legal"].find_impl = lambda _q, _proj=None: [{"chunk_text": "chunk-only sentence"}]
    answers = [
        _llm(policy_quote="We honor deletion requests."),
        _llm(policy_quote="We   honor\ndeletion   requests."),
        _llm(policy_quote="not in the policy", conflict_description="   "),
        _llm(status="conflict", policy_quote="not in the policy", conflict_description="Conflicts."),
        _llm(status="partial", policy_quote="not in the policy", conflict_description="Partial."),
        _llm(status="ambiguous", policy_quote="not in the policy", conflict_description="Ambiguous."),
        _llm(status="missing", policy_quote=None, conflict_description="   "),
        _llm(policy_quote="not in the policy", _analysis_failed=True, conflict_description="Failed."),
        _llm(policy_quote=""),
        _llm(policy_quote="   "),
        _llm(policy_quote="chunk-only sentence"),
    ]
    service, llm = _service(mongo, answers)

    response = _run(service.run(_req()))
    by_id = {gap.statute_chunk_id: gap for gap in response.gaps}

    assert by_id["exact"].status == "addressed"
    assert by_id["exact"].citation_binding_failed is None
    assert by_id["collapsed"].status == "addressed"
    assert by_id["collapsed"].citation_binding_failed is None
    assert by_id["unbound-addressed"].status == "missing"
    assert by_id["unbound-addressed"].policy_quote is None
    assert by_id["unbound-addressed"].citation_binding_failed is True
    assert by_id["unbound-addressed"].conflict_description.startswith("The policy does not contain provisions")
    assert by_id["conflict"].status == "conflict"
    assert by_id["conflict"].policy_quote == "not in the policy"
    assert by_id["conflict"].citation_binding_failed is True
    assert by_id["partial"].status == "partial"
    assert by_id["partial"].citation_binding_failed is True
    assert by_id["ambiguous"].status == "ambiguous"
    assert by_id["ambiguous"].policy_quote == "not in the policy"
    assert by_id["missing"].status == "missing"
    assert by_id["missing"].citation_binding_failed is None
    assert by_id["missing"].conflict_description.startswith("The policy does not contain provisions")
    assert by_id["failed"].status == "addressed"
    assert by_id["failed"].analysis_failed is True
    assert by_id["failed"].citation_binding_failed is None
    assert by_id["failed"].policy_quote == "not in the policy"
    assert by_id["blank-quote"].status == "addressed"
    assert by_id["blank-quote"].policy_quote is None
    assert by_id["blank-quote"].citation_binding_failed is None
    assert by_id["whitespace-quote"].status == "addressed"
    assert by_id["whitespace-quote"].policy_quote == "   "
    assert by_id["whitespace-quote"].citation_binding_failed is None
    assert by_id["chunk-only"].status == "missing"
    assert by_id["chunk-only"].citation_binding_failed is True
    assert response.summary.total_requirements == 11
    assert response.summary.addressed == 4
    assert response.summary.missing == 3
    assert response.summary.conflicts == 1
    assert response.summary.partial == 1
    assert response.summary.ambiguous == 1
    assert response.summary.analysis_failures == 1
    assert llm.gap_check_v4.await_count == 11


def test_empty_policy_text_binds_against_indexed_chunks() -> None:
    mongo, cols = _world()
    _seed_one(cols, policy=_policy(text="   "), statutes=[_statute("hit"), _statute("miss")])
    cols["legal"].find_impl = lambda _q, _proj=None: [{"chunk_text": "Indexed only."}]
    service, _llm_client = _service(
        mongo,
        [
            _llm(policy_quote="Indexed only."),
            _llm(policy_quote="missing from chunks"),
        ],
    )

    response = _run(service.run(_req()))
    by_id = {gap.statute_chunk_id: gap for gap in response.gaps}

    assert by_id["hit"].status == "addressed"
    assert by_id["hit"].citation_binding_failed is None
    assert by_id["miss"].status == "missing"
    assert by_id["miss"].citation_binding_failed is True


def test_policy_chunk_filter_is_equality_for_one_id_and_in_for_many() -> None:
    mongo, cols = _world()
    docs = {
        "pol-1": _policy("pol-1", company="One"),
        "pol-2": _policy("pol-2", company="Two"),
    }

    def find_one(query: Dict[str, Any]):
        return docs.get(query.get("document_id"))

    cols["policies"].find_one_impl = find_one
    cols["legal"].count_impl = lambda _q: 1
    cols["mapping"].find_impl = lambda _q, _proj=None: [_mapping()]
    cols["statutes"].find_impl = lambda _q, _proj=None: [_statute()]
    cols["legal"].find_impl = lambda _q, _proj=None: [{"chunk_text": "We honor deletion requests."}]
    service, _llm_client = _service(mongo, _llm())

    _run(service.run(_req()))
    one_filter = cols["legal"].finds[-1][0]
    assert one_filter["document_id"] == "pol-1"
    assert one_filter["category"] == {"$in": ["rights"]}

    cols["legal"].finds.clear()
    _run(service.run(_req(policy_document_id=None, policy_document_ids=["pol-1", "pol-2"])))
    many_filter = cols["legal"].finds[-1][0]
    assert many_filter["document_id"] == {"$in": ["pol-1", "pol-2"]}
    assert many_filter["category"] == {"$in": ["rights"]}


def test_guards_run_before_mongo_index_and_invalid_status_does_not_persist() -> None:
    mongo, cols = _world()
    _seed_one(cols)
    service, llm = _service(mongo, _llm(), allow=False)
    with pytest.raises(GapAnalysisServiceV4Error) as limited:
        _run(service.run(_req()))
    assert limited.value.status_code == 429
    assert cols["policies"].find_ones == []
    assert llm.gap_check_v4.await_count == 0

    bare, _bare_llm = _service(MagicMock(), _llm())
    with pytest.raises(GapAnalysisServiceV4Error) as missing_ids:
        _run(bare.run(GapAnalysisRequest.model_construct()))
    assert missing_ids.value.status_code == 400
    bare_limiter = bare.rate_limiter
    assert bare_limiter.allow.await_count == 0

    missing_mongo, missing_cols = _world()
    missing_service, _missing_llm = _service(missing_mongo, _llm())
    with pytest.raises(GapAnalysisServiceV4Error) as missing:
        _run(missing_service.run(_req(policy_document_id="missing-pol")))
    assert missing.value.status_code == 404
    assert "missing-pol" in str(missing.value)
    assert missing_cols["legal"].counts == []

    bad, bad_cols = _world()
    _seed_one(bad_cols)
    bad_service, _bad_llm = _service(bad, _llm(status="Addressed"))
    with pytest.raises(ValidationError):
        _run(bad_service.run(_req()))
    assert bad_cols["results"].inserts == []


def test_single_id_list_omits_ids_and_save_false_skips_writes() -> None:
    mongo, cols = _world()
    _seed_one(cols)
    service, _llm_client = _service(mongo, _llm())

    listed = _run(service.run(_req(policy_document_id=None, policy_document_ids=["pol-1"])))
    assert listed.policy_document_id == "pol-1"
    assert listed.policy_document_ids is None
    assert "policy_document_ids" not in cols["results"].inserts[0]
    assert "policy_document_ids" not in cols["logs"].inserts[0]

    cols["results"].inserts.clear()
    cols["logs"].inserts.clear()
    unsaved = _run(service.run(_req(save_results=False)))
    assert unsaved.gaps[0].status == "addressed"
    assert cols["results"].inserts == []
    assert cols["logs"].inserts == []


def test_policy_id_fallback_and_chunk_assembly() -> None:
    mongo, cols = _world()
    calls: List[Dict[str, Any]] = []
    raw_id = "not-an-objectid"
    stored = _policy(
        raw_id,
        text="",
        policy_chunks=[{"chunk_text": "Alpha"}, "skip-me", {"chunk_text": "  "}, {"other": 1}],
    )

    def find_one(query: Dict[str, Any]):
        calls.append(query)
        if query.get("_id") == raw_id:
            return stored
        return None

    cols["policies"].find_one_impl = find_one
    cols["legal"].count_impl = lambda _q: 1
    cols["mapping"].find_impl = lambda _q, _proj=None: [_mapping()]
    cols["statutes"].find_impl = lambda _q, _proj=None: [_statute()]
    cols["legal"].find_impl = lambda _q, _proj=None: []
    service, _llm_client = _service(mongo, _llm(policy_quote="Alpha"))

    response = _run(service.run(_req(policy_document_id=raw_id)))

    assert calls == [{"document_id": raw_id}, {"_id": raw_id}]
    assert response.gaps[0].status == "addressed"
    assert response.gaps[0].citation_binding_failed is None

    hex_id = "a" * 24
    hex_calls: List[Dict[str, Any]] = []

    def find_hex(query: Dict[str, Any]):
        hex_calls.append(query)
        return None

    cols["policies"].find_one_impl = find_hex
    with pytest.raises(GapAnalysisServiceV4Error) as missing:
        _run(service.run(_req(policy_document_id=hex_id)))
    assert missing.value.status_code == 404
    assert len(hex_calls) == 2
    assert isinstance(hex_calls[1]["_id"], ObjectId)
    assert str(hex_calls[1]["_id"]) == hex_id


def test_mappings_without_policy_categories_or_outside_analyze_set_are_skipped() -> None:
    mongo, cols = _world()
    _seed_one(cols)
    cols["mapping"].find_impl = lambda _q, _proj=None: [
        _mapping("definitions", ["defs"]),
        _mapping("consumer_rights", []),
        _mapping("consumer_rights", None),
        _mapping("consumer_rights", [""], sub_topic="blank-cat"),
        _mapping("controller_duties", ["duties"], sub_topic=""),
    ]
    seen_filters: List[Dict[str, Any]] = []

    def statute_find(query: Dict[str, Any], _proj=None):
        seen_filters.append(query)
        return [_statute(query.get("sub_topic") or query["category"])]

    cols["statutes"].find_impl = statute_find
    policy_filters: List[Dict[str, Any]] = []

    def legal_find(query: Dict[str, Any], _proj=None):
        policy_filters.append(query)
        return [{"chunk_text": "We honor deletion requests."}]

    cols["legal"].find_impl = legal_find
    service, llm = _service(mongo, _llm())

    response = _run(service.run(_req()))

    assert seen_filters == [
        {"category": "consumer_rights", "sub_topic": "blank-cat"},
        {"category": "controller_duties"},
    ]
    assert policy_filters[0]["category"] == {"$in": [""]}
    assert policy_filters[1]["category"] == {"$in": ["duties"]}
    assert llm.gap_check_v4.await_count == 2
    assert len(response.gaps) == 2


def test_whitespace_requirement_summary_does_not_fall_back_and_ids_cap_at_500() -> None:
    mongo, cols = _world()
    _seed_one(cols, statutes=[_statute(requirement_summary="Stored summary", subtopic_text="Body")])
    service, _llm_client = _service(mongo, _llm(requirement_summary="   ", confidence="HIGH"))

    padded = _run(service.run(_req(save_results=False)))

    assert padded.gaps[0].requirement_summary == ""
    assert padded.gaps[0].confidence is None

    many, many_cols = _world()
    statutes = [_statute(f"s{i}") for i in range(501)]
    _seed_one(many_cols, statutes=statutes)
    many_service, many_llm = _service(many, _llm(status="missing", policy_quote=None))

    capped = _run(many_service.run(_req()))

    assert many_llm.gap_check_v4.await_count == 501
    assert len(capped.gaps) == 501
    assert capped.retrieval_metadata.statute_items_considered == 501
    assert len(capped.statute_chunk_ids_used) == 500
    assert capped.statute_chunk_ids_used[0] == "s0"
    assert capped.statute_chunk_ids_used[-1] == "s499"
    assert len(many_cols["logs"].inserts[0]["statute_item_ids"]) == 500


def test_missing_jurisdiction_falls_back_and_persist_errors_are_swallowed() -> None:
    mongo, cols = _world()
    _seed_one(cols, statutes=[_statute(jurisdiction="")])
    service, _llm_client = _service(mongo, _llm())

    response = _run(service.run(_req(applicable_jurisdictions=["VA", "CO"], save_results=False)))

    assert response.gaps[0].jurisdiction == "VA"

    cols["statutes"].find_impl = lambda _q, _proj=None: [_statute(jurisdiction=None)]
    omitted = _run(service.run(_req(applicable_jurisdictions=["CO"], save_results=False)))
    assert omitted.gaps[0].jurisdiction == "CO"

    def explode(_doc: Dict[str, Any]) -> None:
        raise RuntimeError("disk full")

    cols["results"].insert_one = explode
    saved = _run(service.run(_req(applicable_jurisdictions=["CA"])))
    assert saved.gaps[0].status == "addressed"
    assert cols["logs"].inserts == []
