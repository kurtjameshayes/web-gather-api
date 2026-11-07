"""Newest drift checkpoint wins, including when it cannot be parsed.

Coverage of the single-row parser (Zulu timestamps, blank values, datetimes,
and the insert shape) already exists. These tests pin the fall-through that
parser does not do: a bad newest row hides older valid rows, trailing space
is not stripped, and an omitted write time uses utc_now.
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

    def __getitem__(self, _name: str):
        run_log = self._run_log

        class _Db:
            def __getitem__(self, _coll_name: str):
                return run_log

        return _Db()


def _storage(run_log: _RunLog) -> ComplianceStorage:
    return ComplianceStorage(_Mongo(run_log), load_config())


def test_unparsable_newest_checkpoint_hides_older_rows() -> None:
    run_log = _RunLog()
    storage = _storage(run_log)
    run_log.docs.extend(
        [
            {
                "type": "drift_meta",
                "last_drift_check_at": "2024-06-01T00:00:00+00:00",
            },
            {
                "type": "other",
                "last_drift_check_at": "2026-01-01T00:00:00+00:00",
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


def test_checkpoint_timestamp_is_not_stripped() -> None:
    run_log = _RunLog()
    storage = _storage(run_log)
    run_log.docs.append(
        {
            "type": "drift_meta",
            "last_drift_check_at": "2026-03-01T00:00:00Z ",
        }
    )

    assert asyncio.run(storage.get_last_drift_check_at()) is None


def test_omitted_checkpoint_time_uses_utc_now() -> None:
    run_log = _RunLog()
    storage = _storage(run_log)
    fixed = datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc)

    with patch("compliance_storage.utc_now", return_value=fixed) as utc_now:
        asyncio.run(storage.set_last_drift_check_at())
        utc_now.assert_called_once()

    with patch("compliance_storage.utc_now") as utc_now:
        asyncio.run(storage.set_last_drift_check_at(fixed))
        utc_now.assert_not_called()

    assert run_log.docs[0]["last_drift_check_at"] == "2026-05-06T07:08:09+00:00"
    assert run_log.docs[1]["last_drift_check_at"] == "2026-05-06T07:08:09+00:00"
