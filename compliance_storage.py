"""Storage for compliance suite results, alerts, and run log."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
