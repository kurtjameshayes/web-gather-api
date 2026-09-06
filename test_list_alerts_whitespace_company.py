"""ComplianceStorage.list_alerts ignores whitespace-only company_name filters."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from compliance_config import load_config
from compliance_storage import ComplianceStorage


def test_list_alerts_whitespace_company_name_is_not_a_filter() -> None:
    """'   ' is truthy but strip() is empty, so no company_name regex is applied."""
    mock_mongo = MagicMock()
    coll = MagicMock()
    coll.count_documents.return_value = 0
    coll.find.return_value.sort.return_value.skip.return_value.limit.return_value = []
    mock_mongo.__getitem__.return_value.__getitem__.return_value = coll

    storage = ComplianceStorage(mock_mongo, load_config())
    docs, total = asyncio.run(storage.list_alerts(company_name="   "))

    assert docs == []
    assert total == 0
    query = coll.count_documents.call_args[0][0]
    assert "company_name" not in query
