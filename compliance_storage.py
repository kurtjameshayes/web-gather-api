"""Storage for compliance suite results, alerts, and run log."""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from compliance_config import ComplianceConfig
from compliance_utils import utc_now


async def _run_in_thread(func, *args, **kwargs):
    """Run sync function in a thread (Python 3.8 compat: asyncio.to_thread added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _iso(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


class ComplianceStorage:
    def __init__(self, mongo_client: Any, config: ComplianceConfig) -> None:
        self._client = mongo_client
        self._config = config
        self._db_name = config.compliance_database
        self._results_coll = config.compliance_results_collection
        self._alerts_coll = config.compliance_alerts_collection
        self._run_log_coll = config.compliance_run_log_collection

    def _db(self):
        return self._client[self._db_name]

    def _results(self):
        return self._db()[self._results_coll]

    def _alerts(self):
        return self._db()[self._alerts_coll]

    def _run_log(self):
        return self._db()[self._run_log_coll]

    async def write_compliance_result(self, doc: Dict[str, Any]) -> str:
        """Write a compliance result (gap analysis and/or health score). Returns inserted id."""
        doc.setdefault("analyzed_at", _iso())
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())

        def insert():
            self._results().insert_one(doc)
            return doc["_id"]

        return await _run_in_thread(insert)

    async def get_last_compliance_result(
        self,
        policy_document_id: str,
        statute_index_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent compliance result for a policy (optionally for a given statute index version)."""
        query: Dict[str, Any] = {"policy_document_id": policy_document_id}
        if statute_index_version is not None:
            query["statute_index_version"] = statute_index_version

        def find():
            cursor = (
                self._results()
                .find(query)
                .sort("analyzed_at", -1)
                .limit(1)
            )
            return next(cursor, None)

        doc = await _run_in_thread(find)
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def write_compliance_alert(self, doc: Dict[str, Any]) -> str:
        """Write a drift alert. Returns alert_id."""
        alert_id = doc.get("alert_id") or str(uuid.uuid4())
        doc["alert_id"] = alert_id
        doc.setdefault("detected_at", _iso())
        doc.setdefault("type", "regulatory_drift")

        def insert():
            self._alerts().insert_one(doc)
            return alert_id

        return await _run_in_thread(insert)

    async def write_compliance_run_log(self, doc: Dict[str, Any]) -> None:
        """Append an audit run log entry."""
        doc.setdefault("run_timestamp", _iso())

        def insert():
            self._run_log().insert_one(doc)

        await _run_in_thread(insert)

    async def get_last_drift_check_at(self) -> Optional[datetime]:
        """Return the timestamp of the last drift check, or None."""
        def find():
            doc = (
                self._run_log()
                .find_one({"type": "drift_meta"}, sort=[("last_drift_check_at", -1)])
            )
            return doc

        doc = await _run_in_thread(find)
        if not doc or not doc.get("last_drift_check_at"):
            return None
        val = doc["last_drift_check_at"]
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    async def set_last_drift_check_at(self, at: Optional[datetime] = None) -> None:
        """Record that a drift check completed at the given time (default now)."""
        now = at or utc_now()
        doc = {
            "type": "drift_meta",
            "last_drift_check_at": _iso(now),
        }

        def insert():
            self._run_log().insert_one(doc)

        await _run_in_thread(insert)

    async def list_policy_document_ids(self, database: Optional[str] = None) -> List[str]:
        """List policy document ids from the policies collection (for drift: policies to re-analyze)."""
        db_name = database or self._db_name
        coll_name = self._config.policies_collection
        id_field = "_id"

        def find():
            cursor = self._client[db_name][coll_name].find({}, {id_field: 1})
            return [str(doc[id_field]) for doc in cursor if doc.get(id_field)]

        return await _run_in_thread(find)

    async def list_runs(
        self,
        policy_document_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        types: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List compliance result runs with filters. Returns (list of run summary dicts, total count)."""
        query: Dict[str, Any] = {}
        if policy_document_id:
            query["policy_document_id"] = policy_document_id
        if types:
            query["run_types"] = {"$in": types}
        if since or until:
            date_expr: Dict[str, Any] = {}
            if since:
                date_expr["$gte"] = since
            if until:
                date_expr["$lte"] = until
            query["analyzed_at"] = date_expr

        def find_and_count():
            coll = self._results()
            total = coll.count_documents(query)
            cursor = (
                coll.find(query)
                .sort("analyzed_at", -1)
                .skip(offset)
                .limit(limit)
            )
            docs = list(cursor)
            return docs, total

        docs, total = await _run_in_thread(find_and_count)
        run_summaries: List[Dict[str, Any]] = []
        for doc in docs:
            run_id = str(doc.get("_id", ""))
            run_at = doc.get("analyzed_at", "")
            type_list: List[str] = doc.get("run_types") or []
            if not type_list:
                if doc.get("gaps") is not None:
                    type_list.append("gap")
                if doc.get("privacy_health_score") is not None:
                    type_list.append("health_score")
                if doc.get("strictest_common_denominator") is not None:
                    type_list.append("multi_jurisdictional")
                if doc.get("applicable_jurisdictions") is not None and not type_list:
                    type_list.append("applicability")
            run_summaries.append({
                "run_id": run_id,
                "policy_document_id": doc.get("policy_document_id", ""),
                "company_name": doc.get("company_name"),
                "run_at": run_at,
                "types": type_list,
                "privacy_health_score": doc.get("privacy_health_score"),
                "score_assessment": doc.get("score_assessment"),
                "summary": doc.get("summary"),
            })
        return run_summaries, total

    async def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a single compliance result by run_id (_id). Returns None if not found."""
        def find():
            return self._results().find_one({"_id": run_id})

        doc = await _run_in_thread(find)
        if not doc:
            return None
        if "_id" in doc:
            doc["run_id"] = str(doc["_id"])
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_alerts(
        self,
        policy_document_id: Optional[str] = None,
        company_name: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List drift alerts with filters. Returns (list of alert dicts, total count)."""
        query: Dict[str, Any] = {}
        if policy_document_id:
            query["policy_document_id"] = policy_document_id
        if company_name and company_name.strip():
            query["company_name"] = {"$regex": re.escape(company_name.strip()), "$options": "i"}
        if jurisdiction:
            query["affected_jurisdictions"] = jurisdiction
        if since:
            query["detected_at"] = {"$gte": since}

        def find_and_count():
            coll = self._alerts()
            total = coll.count_documents(query)
            cursor = (
                coll.find(query)
                .sort("detected_at", -1)
                .skip(offset)
                .limit(limit)
            )
            return list(cursor), total

        docs, total = await _run_in_thread(find_and_count)
        out: List[Dict[str, Any]] = []
        for doc in docs:
            d = dict(doc)
            if "_id" in d:
                del d["_id"]
            out.append(d)
        return out, total
