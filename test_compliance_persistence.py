"""Regression tests for compliance persistence and workflow side effects."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, MagicMock

from cryptography.fernet import Fernet

from audit_logger import AuditLogger
from compliance_config import load_config
from compliance_graph import END, START, StateGraph
from compliance_job_service import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    build_gap_analysis_graph,
)
from compliance_storage import ComplianceStorage
from compliance_suite_schemas import GapAnalysisResponse, GapSummary, RetrievalMetadata
from compliance_utils import hash_text


class _IteratorCursor:
    def __init__(self, docs: Iterable[Dict[str, Any]]) -> None:
        self._docs = iter(docs)

    def __iter__(self) -> "_IteratorCursor":
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self._docs)


def _mongo_with_collection(database: str, collection_name: str, collection: MagicMock) -> MagicMock:
    mongo = MagicMock()
    db = MagicMock()

    def get_db(name: str) -> MagicMock:
        assert name == database
        return db

    def get_collection(name: str) -> MagicMock:
        assert name == collection_name
        return collection

    mongo.__getitem__.side_effect = get_db
    db.__getitem__.side_effect = get_collection
    return mongo


def test_audit_logger_stores_hashes_without_raw_plaintext_when_key_missing(monkeypatch) -> None:
    """Audit logs should remain useful without retaining sensitive policy text."""
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    config = replace(
        load_config(),
        audit_collection="audit_records",
        enable_audit_logging=True,
        allow_raw_audit=True,
    )
    collection = MagicMock()
    mongo = _mongo_with_collection("tenant-db", config.audit_collection, collection)

    asyncio.run(
        AuditLogger(mongo, config).log(
            database="tenant-db",
            policy_id="policy-1",
            jurisdiction="CA",
            policy_text="Sensitive policy text with customer details.",
            sections=[{"section_text": "Consumers may request deletion."}],
            summary={"missing": 0},
        )
    )

    collection.insert_one.assert_called_once()
    record = collection.insert_one.call_args[0][0]
    assert record["policy_id"] == "policy-1"
    assert record["jurisdiction"] == "CA"
    assert record["policy_hash"] == hash_text("Sensitive policy text with customer details.")
    assert record["section_hashes"] == [hash_text("Consumers may request deletion.")]
    assert "encrypted_record" not in record
    assert "encrypted_payload" not in record
    assert "Sensitive policy text" not in json.dumps(record, default=str)


def test_audit_logger_encrypts_raw_payload_only_when_allowed(monkeypatch) -> None:
    """Raw audit payloads must be encrypted and decryptable only through AUDIT_LOG_KEY."""
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    config = replace(
        load_config(),
        audit_collection="audit_records",
        enable_audit_logging=True,
        allow_raw_audit=True,
    )
    collection = MagicMock()
    mongo = _mongo_with_collection("tenant-db", config.audit_collection, collection)

    asyncio.run(
        AuditLogger(mongo, config).log(
            database="tenant-db",
            policy_id="policy-2",
            jurisdiction="VA",
            policy_text="Raw policy text to protect.",
            sections=[{"section_text": "Access rights section."}],
            summary={"addressed": 1},
        )
    )

    record = collection.insert_one.call_args[0][0]
    serialized_record = json.dumps(record, default=str)
    assert "Raw policy text to protect." not in serialized_record
    assert "Access rights section." not in serialized_record
    assert "encrypted_record" in record
    assert "encrypted_payload" in record

    fernet = Fernet(key)
    encrypted_record = json.loads(fernet.decrypt(record["encrypted_record"].encode("utf-8")))
    encrypted_payload = json.loads(fernet.decrypt(record["encrypted_payload"].encode("utf-8")))
    assert encrypted_record["policy_hash"] == hash_text("Raw policy text to protect.")
    assert "policy_text" not in encrypted_record
    assert encrypted_payload == {
        "policy_text": "Raw policy text to protect.",
        "sections": [{"section_text": "Access rights section."}],
    }


def test_compliance_storage_writes_and_reads_latest_result() -> None:
    """Compliance results need stable ids/timestamps and version-aware latest lookup."""
    config = load_config()
    results_collection = MagicMock()
    mongo = _mongo_with_collection(
        config.compliance_database,
        config.compliance_results_collection,
        results_collection,
    )
    storage = ComplianceStorage(mongo, config)
    doc = {
        "policy_document_id": "policy-1",
        "statute_index_version": "idx-2026-05-09",
        "gaps": [],
        "run_types": ["gap"],
    }

    inserted_id = asyncio.run(storage.write_compliance_result(doc))

    assert inserted_id == doc["_id"]
    results_collection.insert_one.assert_called_once_with(doc)
    assert doc["policy_document_id"] == "policy-1"
    assert len(doc["_id"]) == 36
    assert "analyzed_at" in doc

    stored_doc = {
        "_id": inserted_id,
        "policy_document_id": "policy-1",
        "statute_index_version": "idx-2026-05-09",
        "analyzed_at": doc["analyzed_at"],
    }
    find_result = MagicMock()
    sort_result = MagicMock()
    results_collection.find.return_value = find_result
    find_result.sort.return_value = sort_result
    sort_result.limit.return_value = _IteratorCursor([stored_doc])

    latest = asyncio.run(
        storage.get_last_compliance_result(
            "policy-1",
            statute_index_version="idx-2026-05-09",
        )
    )

    results_collection.find.assert_called_once_with(
        {
            "policy_document_id": "policy-1",
            "statute_index_version": "idx-2026-05-09",
        }
    )
    find_result.sort.assert_called_once_with("analyzed_at", -1)
    sort_result.limit.assert_called_once_with(1)
    assert latest == stored_doc


def test_compliance_graph_executes_sync_and_async_nodes_in_order() -> None:
    """The local LangGraph fallback must not stop multi-step workflows early."""
    calls: List[str] = []

    graph = StateGraph(dict)

    def first_node(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(f"first:{state['seed']}")
        return {"first": True}

    async def second_node(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(f"second:{state['first']}")
        return {"second": state["seed"] + "-done"}

    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    result = asyncio.run(graph.compile().ainvoke({"seed": "job"}))

    assert calls == ["first:job", "second:True"]
    assert result == {"seed": "job", "first": True, "second": "job-done"}


def test_gap_analysis_graph_persists_result_and_run_log_after_success() -> None:
    """Completed async gap jobs should persist evidence only after analysis succeeds."""
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    compliance_storage.write_compliance_run_log = AsyncMock()
    run_gap_analysis = AsyncMock(
        return_value=GapAnalysisResponse(
            policy_document_id="policy-1",
            company_name="Example Co",
            applicable_jurisdictions=["CA"],
            analyzed_at="2026-05-09T10:00:00+00:00",
            gaps=[],
            summary=GapSummary(total_requirements=0),
            retrieval_metadata=RetrievalMetadata(statute_items_considered=3),
            statute_chunk_ids_used=["statute-chunk-1"],
            run_types=["gap_v4"],
            run_type="gap_analysis_v4",
            version="v4",
        )
    )

    graph = build_gap_analysis_graph(run_gap_analysis, job_storage, compliance_storage)
    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-1",
                "job_type": "gap_analysis",
                "status": JOB_STATUS_PENDING,
                "request": {
                    "policy_document_id": "policy-1",
                    "applicable_jurisdictions": ["CA"],
                    "run_async": False,
                },
            }
        )
    )

    assert result["status"] == JOB_STATUS_COMPLETED
    assert result["result"]["policy_document_id"] == "policy-1"
    assert result["result"]["run_type"] == "gap_analysis_v4"
    run_gap_analysis.assert_awaited_once()
    assert run_gap_analysis.await_args.args[0].policy_document_id == "policy-1"

    status_updates = [
        call.args[1]
        for call in job_storage.update_job_status.call_args_list
    ]
    assert status_updates == [JOB_STATUS_RUNNING, JOB_STATUS_COMPLETED]

    compliance_storage.write_compliance_result.assert_awaited_once()
    result_doc = compliance_storage.write_compliance_result.await_args.args[0]
    assert result_doc == {
        "policy_document_id": "policy-1",
        "company_name": "Example Co",
        "applicable_jurisdictions": ["CA"],
        "gaps": [],
        "summary": {"total_requirements": 0, "missing": 0, "addressed": 0, "conflicts": 0, "partial": 0, "ambiguous": 0, "analysis_failures": 0},
        "analyzed_at": "2026-05-09T10:00:00+00:00",
        "run_types": ["gap_v4"],
        "run_type": "gap_analysis_v4",
        "version": "v4",
        "retrieval_metadata": {
            "statute_subchunks_considered": 0,
            "statute_chunks_considered": 0,
            "statute_items_considered": 3,
            "statute_pairs_matched": 0,
        },
    }

    compliance_storage.write_compliance_run_log.assert_awaited_once()
    run_log = compliance_storage.write_compliance_run_log.await_args.args[0]
    assert run_log == {
        "policy_document_id": "policy-1",
        "applicable_jurisdictions": ["CA"],
        "run_timestamp": "2026-05-09T10:00:00+00:00",
        "statute_chunk_ids_used": ["statute-chunk-1"],
        "run_type": "gap_analysis_v4",
        "summary": {"total_requirements": 0, "missing": 0, "addressed": 0, "conflicts": 0, "partial": 0, "ambiguous": 0, "analysis_failures": 0},
    }


def test_gap_analysis_graph_marks_job_failed_without_persisting_on_error() -> None:
    """Failed gap jobs should be terminal and must not write compliance evidence."""
    job_storage = MagicMock()
    compliance_storage = MagicMock()
    compliance_storage.write_compliance_result = AsyncMock()
    compliance_storage.write_compliance_run_log = AsyncMock()
    run_gap_analysis = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    graph = build_gap_analysis_graph(run_gap_analysis, job_storage, compliance_storage)
    result = asyncio.run(
        graph.ainvoke(
            {
                "job_id": "job-2",
                "job_type": "gap_analysis",
                "status": JOB_STATUS_PENDING,
                "request": {
                    "policy_document_id": "policy-2",
                    "run_async": False,
                },
            }
        )
    )

    assert result["status"] == JOB_STATUS_FAILED
    assert result["error"] == "LLM unavailable"
    status_updates = [
        call.args[1]
        for call in job_storage.update_job_status.call_args_list
    ]
    assert status_updates == [JOB_STATUS_RUNNING, JOB_STATUS_FAILED]
    compliance_storage.write_compliance_result.assert_not_awaited()
    compliance_storage.write_compliance_run_log.assert_not_awaited()
