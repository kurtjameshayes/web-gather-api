"""Regression tests for audit logging privacy and async thread helpers."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from audit_logger import AuditLogger
from compliance_config import load_config
from compliance_utils import hash_text


def _config(enable: bool = True, allow_raw: bool = False):
    config = load_config()
    config.enable_audit_logging = enable
    config.allow_raw_audit = allow_raw
    config.audit_collection = "audit_records"
    return config


def _log_once(logger: AuditLogger) -> None:
    asyncio.run(
        logger.log(
            database="privacy-db",
            policy_id="policy-1",
            jurisdiction="CA",
            policy_text="We collect email addresses for account support.",
            sections=[{"section_text": "Businesses must disclose data collection."}],
            summary={"status": "non_compliant"},
        )
    )


def test_audit_logging_disabled_skips_database_access(monkeypatch):
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    mongo = MagicMock()
    logger = AuditLogger(mongo, _config(enable=False))

    _log_once(logger)

    mongo.__getitem__.assert_not_called()


def test_audit_logger_without_key_stores_hashes_only(monkeypatch):
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    collection = MagicMock()
    mongo = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = collection
    logger = AuditLogger(mongo, _config(enable=True, allow_raw=True))

    with patch("audit_logger.run_in_thread", side_effect=_sync_run):
        _log_once(logger)

    collection.insert_one.assert_called_once()
    record = collection.insert_one.call_args[0][0]
    assert record["policy_hash"] == hash_text("We collect email addresses for account support.")
    assert record["section_hashes"] == [hash_text("Businesses must disclose data collection.")]
    assert record["summary"] == {"status": "non_compliant"}
    assert "encrypted_record" not in record
    assert "encrypted_payload" not in record
    assert "policy_text" not in record
    assert "sections" not in record


def test_audit_logger_encrypts_raw_payload_only_when_key_allows(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    collection = MagicMock()
    mongo = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = collection
    logger = AuditLogger(mongo, _config(enable=True, allow_raw=True))

    with patch("audit_logger.run_in_thread", side_effect=_sync_run):
        _log_once(logger)

    record = collection.insert_one.call_args[0][0]
    assert "policy_text" not in record
    assert "sections" not in record
    encrypted_record = json.loads(Fernet(key).decrypt(record["encrypted_record"]).decode("utf-8"))
    encrypted_payload = json.loads(Fernet(key).decrypt(record["encrypted_payload"]).decode("utf-8"))
    assert encrypted_record["policy_hash"] == record["policy_hash"]
    assert encrypted_record["section_hashes"] == record["section_hashes"]
    assert encrypted_payload == {
        "policy_text": "We collect email addresses for account support.",
        "sections": [{"section_text": "Businesses must disclose data collection."}],
    }


def test_run_in_thread_falls_back_when_to_thread_unavailable(monkeypatch):
    import asyncio as asyncio_module
    import async_utils

    monkeypatch.delattr(asyncio_module, "to_thread", raising=False)

    result = asyncio.run(async_utils.run_in_thread(lambda x, *, y: x + y, 2, y=3))

    assert result == 5


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)
