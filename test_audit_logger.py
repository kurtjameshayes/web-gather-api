"""Audit logger persistence and encryption contracts."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from audit_logger import AuditLogger
from compliance_utils import hash_text


def _config(**overrides) -> SimpleNamespace:
    values = {
        "audit_collection": "policy_compliance_audit",
        "enable_audit_logging": True,
        "allow_raw_audit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def test_audit_logger_disabled_does_not_write() -> None:
    mock_mongo = MagicMock()
    logger = AuditLogger(mock_mongo, _config(enable_audit_logging=False))
    asyncio.run(
        logger.log(
            database="privacy-compliance",
            policy_id="pol-1",
            jurisdiction="CA",
            policy_text="secret policy",
            sections=[{"section_text": "section"}],
            summary={"missing": 1},
        )
    )
    mock_mongo.__getitem__.assert_not_called()


def test_audit_logger_writes_hashes_not_raw_text() -> None:
    mock_mongo = MagicMock()
    collection = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection
    logger = AuditLogger(mock_mongo, _config())

    with patch("audit_logger.run_in_thread", side_effect=_sync_run):
        asyncio.run(
            logger.log(
                database="privacy-compliance",
                policy_id="pol-1",
                jurisdiction="CA",
                policy_text="secret policy",
                sections=[{"section_text": "section body"}],
                summary={"missing": 1},
            )
        )

    collection.insert_one.assert_called_once()
    record = collection.insert_one.call_args[0][0]
    assert record["policy_id"] == "pol-1"
    assert record["jurisdiction"] == "CA"
    assert record["policy_hash"] == hash_text("secret policy")
    assert record["section_hashes"] == [hash_text("section body")]
    assert "policy_text" not in record
    assert "encrypted_record" not in record
    assert "encrypted_payload" not in record


def test_audit_logger_encrypts_record_when_key_set(monkeypatch) -> None:
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    mock_mongo = MagicMock()
    collection = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection
    logger = AuditLogger(mock_mongo, _config(allow_raw_audit=True))

    with patch("audit_logger.run_in_thread", side_effect=_sync_run):
        asyncio.run(
            logger.log(
                database="privacy-compliance",
                policy_id="pol-1",
                jurisdiction="CA",
                policy_text="secret policy",
                sections=[{"section_text": "section body"}],
                summary={"missing": 0},
            )
        )

    record = collection.insert_one.call_args[0][0]
    assert "encrypted_record" in record
    assert "encrypted_payload" in record
    fernet = Fernet(key)
    payload = fernet.decrypt(record["encrypted_payload"].encode("utf-8")).decode("utf-8")
    assert "secret policy" in payload
    assert "section body" in payload


def test_audit_logger_allow_raw_without_key_skips_payload() -> None:
    mock_mongo = MagicMock()
    collection = MagicMock()
    mock_mongo.__getitem__.return_value.__getitem__.return_value = collection
    logger = AuditLogger(mock_mongo, _config(allow_raw_audit=True))

    with patch("audit_logger.run_in_thread", side_effect=_sync_run):
        asyncio.run(
            logger.log(
                database="privacy-compliance",
                policy_id="pol-1",
                jurisdiction="CA",
                policy_text="secret policy",
                sections=[{"section_text": "section body"}],
                summary={},
            )
        )

    record = collection.insert_one.call_args[0][0]
    assert "encrypted_payload" not in record
    assert "encrypted_record" not in record
