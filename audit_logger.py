"""Audit logger for compliance comparisons (hashes only)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet

from async_utils import run_in_thread
from compliance_config import ComplianceConfig
from compliance_utils import hash_text, utc_now


class AuditLogger:
    def __init__(self, mongo_client: Any, config: ComplianceConfig) -> None:
        self._mongo_client = mongo_client
        self._collection = config.audit_collection
        self._enabled = config.enable_audit_logging
        self._allow_raw = config.allow_raw_audit
        key = os.getenv("AUDIT_LOG_KEY")
        self._fernet = Fernet(key) if key else None

    async def log(
        self,
        database: str,
        policy_id: Optional[str],
        jurisdiction: str,
        policy_text: str,
        sections: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        if not self._enabled:
            return

        record: Dict[str, Any] = {
            "policy_id": policy_id,
            "jurisdiction": jurisdiction,
            "policy_hash": hash_text(policy_text),
            "section_hashes": [hash_text(section["section_text"]) for section in sections],
            "summary": summary,
            "created_at": utc_now(),
        }

        if self._fernet:
            payload_record = dict(record)
            payload_record["created_at"] = record["created_at"].isoformat()
            record["encrypted_record"] = self._fernet.encrypt(
                json.dumps(payload_record).encode("utf-8")
            ).decode("utf-8")

        if self._allow_raw and self._fernet:
            payload = {
                "policy_text": policy_text,
                "sections": sections,
            }
            record["encrypted_payload"] = self._fernet.encrypt(
                json.dumps(payload).encode("utf-8")
            ).decode("utf-8")

        collection = self._mongo_client[database][self._collection]

        await run_in_thread(lambda: collection.insert_one(record))
