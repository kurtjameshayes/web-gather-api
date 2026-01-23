"""Database-related functions and endpoints.

Includes functionality for endpoints: all-collections, all-databases, collections, databases, and documents.
"""
import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None
WEB_GATHER_DB = "web-gather"
DOCUMENTS_COLLECTION = "documents"
EMBEDDING_MODEL_COLLECTION = "embedding_model"

db_bp = Blueprint("db", __name__)


def init_db(client):
    """Initialize the database module with the MongoDB client."""
    global mongo_client
    mongo_client = client


def utc_now():
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def get_embedding_model_name(database_name: str):
    """Look up the embedding model configured for a database."""
    logger.info("Looking up embedding model for database: %s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    record = wg_db[EMBEDDING_MODEL_COLLECTION].find_one(
        {"database_name": database_name}
    )
    if not record:
        logger.warning("No embedding model configured for database: %s", database_name)
        return None
    model_name = record.get("model_name")
    logger.info("Found embedding model: %s for database: %s", model_name, database_name)
    return model_name


@db_bp.get("/documents")
def list_documents():
    """List uploaded documents for a collection."""
    logger.info("GET /documents - Listing documents")
    database_name = request.args.get("database_name")
    collection_name = request.args.get("collection_name")
    if not database_name or not collection_name:
        logger.warning("GET /documents - Missing required parameters")
        return jsonify({"error": "database_name and collection_name are required"}), 400

    logger.info("GET /documents - Querying %s.%s", database_name, collection_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    docs = list(
        wg_db[DOCUMENTS_COLLECTION].find(
            {"database_name": database_name, "collection_name": collection_name},
            {"_id": 0},
        )
    )
    logger.info("GET /documents - Found %d documents", len(docs))
    return jsonify({"documents": docs})


@db_bp.get("/databases")
def list_databases():
    """List all databases with uploaded documents."""
    logger.info("GET /databases - Listing databases")
    wg_db = mongo_client[WEB_GATHER_DB]
    databases = wg_db[DOCUMENTS_COLLECTION].distinct("database_name")
    logger.info("GET /databases - Found %d databases", len(databases))
    return jsonify({"databases": databases})


@db_bp.get("/all-databases")
def list_all_databases():
    """List all databases in MongoDB."""
    logger.info("GET /all-databases - Listing all MongoDB databases")
    databases = mongo_client.list_database_names()
    logger.info("GET /all-databases - Found %d databases", len(databases))
    return jsonify({"databases": databases})


@db_bp.get("/collections")
def list_collections():
    """List collections with uploaded documents."""
    logger.info("GET /collections - Listing collections")
    database_name = request.args.get("database_name")
    if not database_name:
        logger.warning("GET /collections - Missing required parameter: database_name")
        return jsonify({"error": "database_name is required"}), 400

    logger.info("GET /collections - Querying collections for database: %s", database_name)
    wg_db = mongo_client[WEB_GATHER_DB]
    collections = wg_db[DOCUMENTS_COLLECTION].distinct(
        "collection_name", {"database_name": database_name}
    )
    logger.info("GET /collections - Found %d collections", len(collections))
    return jsonify({"database_name": database_name, "collections": collections})


@db_bp.get("/all-collections")
def list_all_collections():
    """List all collections in a MongoDB database."""
    logger.info("GET /all-collections - Listing all MongoDB collections")
    database_name = request.args.get("database_name")
    if not database_name:
        logger.warning("GET /all-collections - Missing required parameter: database_name")
        return jsonify({"error": "database_name is required"}), 400

    logger.info("GET /all-collections - Querying all collections for database: %s", database_name)
    collections = mongo_client[database_name].list_collection_names()
    logger.info("GET /all-collections - Found %d collections", len(collections))
    return jsonify({"database_name": database_name, "collections": collections})
