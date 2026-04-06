"""Regression tests for audit logging behavior."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from audit_logger import AuditLogger
from compliance_config import load_config
from compliance_utils import hash_text


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_log_returns_early_when_disabled(anyio_backend: str) -> None:
    config = load_config()
    config.enable_audit_logging = False

    mongo = MagicMock()
    logger = AuditLogger(mongo, config)

    await logger.log(
        database="privacy-db",
        policy_id="pol-1",
        jurisdiction="CA",
        policy_text="policy text",
        sections=[{"section_text": "s1"}],
        summary={"missing": 1},
    )

    mongo.__getitem__.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_log_persists_hashes_without_encryption(
    monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True

    fixed_now = datetime(2026, 4, 6, tzinfo=timezone.utc)
    monkeypatch.setattr("audit_logger.utc_now", lambda: fixed_now)

    inserted: dict[str, object] = {}
    collection = MagicMock()
    collection.insert_one.side_effect = lambda doc: inserted.update(doc)
    db = MagicMock()
    db.__getitem__.return_value = collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    logger = AuditLogger(mongo, config)

    await logger.log(
        database="privacy-db",
        policy_id="pol-1",
        jurisdiction="CA",
        policy_text="policy text",
        sections=[{"section_text": "alpha"}, {"section_text": "beta"}],
        summary={"missing": 2},
    )

    assert inserted["policy_id"] == "pol-1"
    assert inserted["jurisdiction"] == "CA"
    assert inserted["summary"] == {"missing": 2}
    assert inserted["created_at"] == fixed_now
    assert inserted["policy_hash"] == hash_text("policy text")
    assert inserted["section_hashes"] == [hash_text("alpha"), hash_text("beta")]
    assert "encrypted_record" not in inserted
    assert "encrypted_payload" not in inserted


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_log_encrypts_record_and_optional_raw_payload(
    monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
    config = load_config()
    config.enable_audit_logging = True
    config.allow_raw_audit = True

    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("AUDIT_LOG_KEY", key)
    fixed_now = datetime(2026, 4, 6, tzinfo=timezone.utc)
    monkeypatch.setattr("audit_logger.utc_now", lambda: fixed_now)

    inserted: dict[str, object] = {}
    collection = MagicMock()
    collection.insert_one.side_effect = lambda doc: inserted.update(doc)
    db = MagicMock()
    db.__getitem__.return_value = collection
    mongo = MagicMock()
    mongo.__getitem__.return_value = db

    logger = AuditLogger(mongo, config)
    fernet = Fernet(key.encode("utf-8"))

    sections = [{"section_text": "alpha"}]
    await logger.log(
        database="privacy-db",
        policy_id="pol-2",
        jurisdiction="VA",
        policy_text="secret body",
        sections=sections,
        summary={"missing": 0},
    )

    assert isinstance(inserted.get("encrypted_record"), str)
    assert isinstance(inserted.get("encrypted_payload"), str)

    decrypted_record = json.loads(
        fernet.decrypt(inserted["encrypted_record"].encode("utf-8")).decode("utf-8")
    )
    assert decrypted_record["created_at"] == fixed_now.isoformat()
    assert decrypted_record["policy_id"] == "pol-2"
    assert decrypted_record["section_hashes"] and len(decrypted_record["section_hashes"]) == 1

    decrypted_payload = json.loads(
        fernet.decrypt(inserted["encrypted_payload"].encode("utf-8")).decode("utf-8")
    )
    assert decrypted_payload == {"policy_text": "secret body", "sections": sections}
