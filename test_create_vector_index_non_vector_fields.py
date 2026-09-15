"""POST /create-vector-index copies non-vector embedding_model.fields through.

Vector entries are forced to cosine (open PR #209). Non-vector dicts and
non-dict field items must still be included in the Atlas definition; dropping
them would silently change filter/index shape for compliance search.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

sys.modules.setdefault("anthropic", MagicMock())

from flask import Flask

from util import init_util, util_bp


def test_create_vector_index_copies_non_vector_fields_and_forces_cosine() -> None:
    target_db = MagicMock()
    target_db.command.side_effect = [
        Exception("index missing"),
        {"ok": 1},
    ]
    mock_mongo = MagicMock()
    mock_mongo.__getitem__.return_value = target_db
    init_util(mock_mongo)

    record = {
        "database_name": "privacy-compliance",
        "model_name": "m",
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "euclidean",
            },
            {"type": "filter", "path": "jurisdiction"},
            "legacy-string-field",
        ],
    }
    with patch("util.get_embedding_model_record", return_value=record):
        app = Flask(__name__)
        app.register_blueprint(util_bp)
        client = app.test_client()
        response = client.post(
            "/create-vector-index",
            json={
                "database_name": "privacy-compliance",
                "collection_name": "policy_embeddings",
                "index_name": "vector_index",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    fields = payload["definition"]["fields"]
    assert fields[0]["type"] == "vector"
    assert fields[0]["similarity"] == "cosine"
    assert fields[0]["path"] == "embedding"
    assert fields[1] == {"type": "filter", "path": "jurisdiction"}
    assert fields[2] == "legacy-string-field"

    create_cmd = target_db.command.call_args_list[1].args[0]
    assert create_cmd["indexes"][0]["definition"]["fields"] == fields
