"""Audit logger must store hashes, and raw policy text only inside an encrypted payload."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from cryptography.fernet import Fernet

from audit_logger import AuditLogger
from compliance_config import load_config
from compliance_utils import hash_text

POLICY_TEXT = "UNIQUE_POLICY_BODY_do_not_store"
SECTION_TEXT = "UNIQUE_SECTION_BODY_do_not_store"


def _storage() -> tuple[MagicMock, MagicMock, dict[str, Any]]:
    captured: dict[str, Any] = {}
    collection = MagicMock()

    def insert_one(record: dict[str, Any]) -> None:
        captured["record"] = record

    collection.insert_one.side_effect = insert_one
    database = MagicMock()
    database.__getitem__.return_value = collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = database
    return mongo, database, captured


def _config(**overrides: Any):
    config = load_config()
    config.audit_collection = "policy_compliance_audit"
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _log(logger: AuditLogger) -> None:
    asyncio.run(
        logger.log(
            "privacy-compliance",
            "policy-1",
            "CA",
            POLICY_TEXT,
            [{"section_text": SECTION_TEXT, "section_id": "retention"}],
            {"gaps": 1},
        )
    )


def test_audit_logger_disabled_does_not_write(monkeypatch: Any) -> None:
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    mongo, _database, captured = _storage()
    logger = AuditLogger(mongo, _config(enable_audit_logging=False, allow_raw_audit=True))
    _log(logger)
    assert captured == {}
    mongo.__getitem__.assert_not_called()


def test_audit_logger_stores_hashes_without_plaintext(monkeypatch: Any) -> None:
    monkeypatch.delenv("AUDIT_LOG_KEY", raising=False)
    mongo, database, captured = _storage()
    logger = AuditLogger(mongo, _config(enable_audit_logging=True, allow_raw_audit=True))
    _log(logger)

    mongo.__getitem__.assert_called_with("privacy-compliance")
    database.__getitem__.assert_called_with("policy_compliance_audit")
    record = captured["record"]
    serialized = json.dumps(record, default=str)
    assert POLICY_TEXT not in serialized
    assert SECTION_TEXT not in serialized
    assert record["policy_id"] == "policy-1"
    assert record["jurisdiction"] == "CA"
    assert record["policy_hash"] == hash_text(POLICY_TEXT)
    assert record["section_hashes"] == [hash_text(SECTION_TEXT)]
    assert record["summary"] == {"gaps": 1}
    assert "encrypted_record" not in record
    assert "encrypted_payload" not in record
    assert "policy_text" not in record


def test_audit_logger_encrypts_record_and_raw_payload(monkeypatch: Any) -> None:
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    fernet = Fernet(key)
    _mongo, _database, captured = _storage()
    logger = AuditLogger(_mongo, _config(enable_audit_logging=True, allow_raw_audit=True))
    _log(logger)

    record = captured["record"]
    assert POLICY_TEXT not in json.dumps(
        {k: v for k, v in record.items() if k not in {"encrypted_record", "encrypted_payload"}},
        default=str,
    )
    decrypted_record = json.loads(fernet.decrypt(record["encrypted_record"].encode("utf-8")))
    assert decrypted_record["policy_hash"] == hash_text(POLICY_TEXT)
    assert "policy_text" not in decrypted_record
    assert SECTION_TEXT not in json.dumps(decrypted_record)

    decrypted_payload = json.loads(fernet.decrypt(record["encrypted_payload"].encode("utf-8")))
    assert decrypted_payload["policy_text"] == POLICY_TEXT
    assert decrypted_payload["sections"][0]["section_text"] == SECTION_TEXT


def test_audit_logger_omits_raw_payload_when_disallowed(monkeypatch: Any) -> None:
    key = Fernet.generate_key()
    monkeypatch.setenv("AUDIT_LOG_KEY", key.decode("utf-8"))
    fernet = Fernet(key)
    _mongo, _database, captured = _storage()
    logger = AuditLogger(_mongo, _config(enable_audit_logging=True, allow_raw_audit=False))
    _log(logger)

    record = captured["record"]
    assert "encrypted_payload" not in record
    decrypted_record = json.loads(fernet.decrypt(record["encrypted_record"].encode("utf-8")))
    assert decrypted_record["policy_hash"] == hash_text(POLICY_TEXT)
    assert POLICY_TEXT not in json.dumps(decrypted_record)
    assert SECTION_TEXT not in json.dumps(record["encrypted_record"])
