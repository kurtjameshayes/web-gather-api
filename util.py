"""Utility endpoints for the Web Gather API.

Includes endpoints: embedding-models POST and GET.
"""
import logging

from flask import Blueprint, jsonify, request

from db import (
    WEB_GATHER_DB,
    EMBEDDING_MODEL_COLLECTION,
    utc_now,
)

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None

util_bp = Blueprint("util", __name__)


def init_util(client):
    """Initialize the util module with the MongoDB client."""
    global mongo_client
    mongo_client = client


@util_bp.get("/embedding-models")
def list_embedding_models():
    """List embedding models."""
    logger.info("GET /embedding-models - Listing embedding models")
    database_name = request.args.get("database_name")
    logger.info("GET /embedding-models - Parameters: database_name=%s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    query = {"database_name": database_name} if database_name else {}
    models = list(wg_db[EMBEDDING_MODEL_COLLECTION].find(query, {"_id": 0}))
    logger.info("GET /embedding-models - Found %d models", len(models))
    return jsonify({"models": models})


@util_bp.post("/embedding-models")
def add_embedding_model():
    """Add embedding model."""
    logger.info("POST /embedding-models - Adding embedding model")
    payload = request.get_json(silent=True) or {}
    database_name = payload.get("database_name")
    model_name = payload.get("model_name")
    logger.info("POST /embedding-models - Parameters: database_name=%s, model_name=%s", database_name, model_name)
    if not database_name or not model_name:
        logger.warning("POST /embedding-models - Missing required parameters")
        return jsonify({"error": "database_name and model_name are required"}), 400

    logger.info("POST /embedding-models - Setting model %s for database %s", model_name, database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    wg_db[EMBEDDING_MODEL_COLLECTION].update_one(
        {"database_name": database_name},
        {
            "$set": {
                "database_name": database_name,
                "model_name": model_name,
                "updated_at": utc_now(),
            }
        },
        upsert=True,
    )

    logger.info("POST /embedding-models - Successfully configured model %s for database %s", model_name, database_name)
    return jsonify({"database_name": database_name, "model_name": model_name})
