"""Pin current create-sub-vector-index source_query $where enqueue contract.

Unlike POST /create-embeddings, this starter currently does not reject dangerous
Mongo operators. A $where dict is forwarded into the background job request.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from core import core_bp, init_core


@pytest.fixture
def app() -> Flask:
    init_core(MagicMock(), MagicMock(), MagicMock())
    app = Flask(__name__)
    app.register_blueprint(core_bp)
    return app


@pytest.fixture
def client(app: Flask):
    return app.test_client()


def test_create_sub_vector_index_enqueues_where_source_query(client) -> None:
    """$where currently starts a 202 job; it is not rejected before enqueue."""
    where_query = {"$where": "1==1"}
    with patch("core._get_index_job_storage", return_value=MagicMock()), patch(
        "index_job_service.start_sub_vector_index_job", return_value="job-where"
    ) as start_job:
        response = client.post(
            "/create-sub-vector-index",
            json={"document_type": "policy", "source_query": where_query},
        )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "job-where"
    request_dict = start_job.call_args.kwargs["request_dict"]
    assert request_dict["document_type"] == "policy"
    assert request_dict["source_query"] == where_query
