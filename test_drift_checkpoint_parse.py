"""Drift checkpoint reads and writes in ComplianceStorage.

The newest drift_meta row is the checkpoint. An unparsable value becomes
None and older valid rows are not consulted, so a corrupt checkpoint
stops new-statute detection instead of falling back.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

from compliance_config import load_config
from compliance_storage import ComplianceStorage


class _RunLog:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    def insert_one(self, doc: dict) -> None:
        self.docs.append(doc)

    def find_one(self, query, sort=None):
        matches = [
            doc
            for doc in self.docs
            if all(doc.get(key) == value for key, value in query.items())
        ]
        if sort:
            for key, direction in reversed(sort):
                matches.sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        return matches[0] if matches else None


class _Mongo:
    def __init__(self, run_log: _RunLog) -> None:
        self._run_log = run_log
        self.names: list[tuple[str, str]] = []

    def __getitem__(self, name: str):
        run_log = self._run_log
        names = self.names

        class _Db:
            def __getitem__(self, coll_name: str):
                names.append((name, coll_name))
                return run_log

        return _Db()


def _storage(run_log: _RunLog) -> tuple[ComplianceStorage, _Mongo]:
    config = load_config()
    mongo = _Mongo(run_log)
    return ComplianceStorage(mongo, config), mongo


def test_checkpoint_round_trip_uses_newest_drift_meta_row() -> None:
    run_log = _RunLog()
    storage, mongo = _storage(run_log)
    earlier = datetime(2020, 1, 1, tzinfo=timezone.utc)
    later = datetime(2024, 6, 1, 12, 30, tzinfo=timezone.utc)

    asyncio.run(storage.set_last_drift_check_at(earlier))
    asyncio.run(storage.set_last_drift_check_at(later))
    run_log.docs.append(
        {
            "type": "other",
            "last_drift_check_at": "2026-01-01T00:00:00+00:00",
        }
    )

    found = asyncio.run(storage.get_last_drift_check_at())

    assert found == later
    assert [doc["type"] for doc in run_log.docs[:2]] == ["drift_meta", "drift_meta"]
    assert run_log.docs[0]["last_drift_check_at"] == "2020-01-01T00:00:00+00:00"
    assert run_log.docs[1]["last_drift_check_at"] == "2024-06-01T12:30:00+00:00"
    assert mongo.names
    assert {db for db, _coll in mongo.names} == {storage._db_name}
    assert {coll for _db, coll in mongo.names} == {storage._config.compliance_run_log_collection}


def test_omitted_checkpoint_time_uses_utc_now() -> None:
    run_log = _RunLog()
    storage, _mongo = _storage(run_log)
    fixed = datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc)

    with patch("compliance_storage.utc_now", return_value=fixed) as utc_now:
        asyncio.run(storage.set_last_drift_check_at())
        utc_now.assert_called_once()

    with patch("compliance_storage.utc_now") as utc_now:
        asyncio.run(storage.set_last_drift_check_at(fixed))
        utc_now.assert_not_called()

    assert run_log.docs[0] == {
        "type": "drift_meta",
        "last_drift_check_at": "2026-05-06T07:08:09+00:00",
    }
    assert run_log.docs[1] == run_log.docs[0]


def test_unparsable_newest_checkpoint_hides_older_rows() -> None:
    run_log = _RunLog()
    storage, _mongo = _storage(run_log)
    run_log.docs.extend(
        [
            {
                "type": "drift_meta",
                "last_drift_check_at": "2024-06-01T00:00:00+00:00",
            },
            {
                "type": "drift_meta",
                "last_drift_check_at": "2024-99-not-a-date",
            },
        ]
    )

    assert asyncio.run(storage.get_last_drift_check_at()) is None

    run_log.docs.append(
        {
            "type": "drift_meta",
            "last_drift_check_at": "2025-01-01T00:00:00+00:00",
        }
    )
    assert asyncio.run(storage.get_last_drift_check_at()) == datetime(
        2025, 1, 1, tzinfo=timezone.utc
    )


def test_checkpoint_parser_accepts_zulu_and_rejects_other_shapes() -> None:
    run_log = _RunLog()
    storage, _mongo = _storage(run_log)

    assert asyncio.run(storage.get_last_drift_check_at()) is None

    aware = datetime(2026, 3, 1, 4, 5, tzinfo=timezone.utc)
    cases = [
        ("2026-03-01T00:00:00Z", datetime(2026, 3, 1, tzinfo=timezone.utc)),
        ("2026-03-01T04:05:00+00:00", aware),
        (aware, aware),
        ("  2026-03-01T00:00:00Z  ", None),
        ("yesterday", None),
        ("", None),
        (None, None),
        (0, None),
        (1_700_000_000, None),
    ]
    for stored, expected in cases:
        run_log.docs[:] = [
            {"type": "drift_meta", "last_drift_check_at": stored},
        ]
        found = asyncio.run(storage.get_last_drift_check_at())
        if expected is None:
            assert found is None
        elif isinstance(stored, datetime):
            assert found is stored
        else:
            assert found == expected

    run_log.docs[:] = [{"type": "drift_meta"}]
    assert asyncio.run(storage.get_last_drift_check_at()) is None
