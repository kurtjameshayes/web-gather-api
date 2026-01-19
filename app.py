import json
import logging
import os
import uuid
from datetime import datetime, timezone

import anthropic
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from firecrawl import FirecrawlApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("web-gather-api")

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not FIRECRAWL_API_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY is not set")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

WEB_GATHER_DB = "web-gather"
DOCUMENTS_COLLECTION = "documents"
EMBEDDING_MODEL_COLLECTION = "embedding_model"

logger.info("Initializing Flask application")
app = Flask(__name__)

logger.info("Connecting to MongoDB at %s", MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else "localhost")
mongo_client = MongoClient(MONGODB_URI)

logger.info("Initializing Firecrawl client")
firecrawl_client = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

logger.info("Initializing Anthropic client")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_model_cache = {}


def build_openapi_spec():
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Web Gather API",
            "version": "1.0.0",
            "description": "Gather, ingest, index, and search web documents.",
        },
        "paths": {
            "/gather": {
                "post": {
                    "summary": "Gather web documents based on query",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Search results"}},
                }
            },
            "/ingest": {
                "post": {
                    "summary": "Ingest a web document by crawling a URL",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "depth": {"type": "integer"},
                                        "breadth": {"type": "integer"},
                                        "database": {"type": "string"},
                                        "collection": {"type": "string"},
                                    },
                                    "required": ["url", "database", "collection"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Ingestion result"}},
                }
            },
            "/documents": {
                "get": {
                    "summary": "List uploaded documents for a collection",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "collection_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Documents list"}},
                }
            },
            "/collections": {
                "get": {
                    "summary": "List collections with uploaded documents",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Collections list"}},
                }
            },
            "/index": {
                "post": {
                    "summary": "Index a document by id",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"document_id": {"type": "string"}},
                                    "required": ["document_id"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Indexing result"}},
                }
            },
            "/search": {
                "get": {
                    "summary": "Search vector-indexed collection",
                    "parameters": [
                        {
                            "name": "document_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "query",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Search results"}},
                }
            },
            "/embedding-models": {
                "get": {
                    "summary": "List embedding models",
                    "parameters": [
                        {
                            "name": "database_name",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Embedding models"}},
                },
                "post": {
                    "summary": "Add embedding model",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "database_name": {"type": "string"},
                                        "model_name": {"type": "string"},
                                    },
                                    "required": ["database_name", "model_name"],
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Embedding model saved"}},
                },
            },
            "/parse_llm": {
                "get": {
                    "summary": "Parse document text using LLM",
                    "description": "Dynamically parse document text according to specific instructions using a language model. Returns structured JSON with parsed sections.",
                    "parameters": [
                        {
                            "name": "document",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "The document text to parse",
                        },
                        {
                            "name": "parsing_prompt",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Instructions for how to parse the document",
                        },
                        {
                            "name": "document_id",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "string"},
                            "description": "Optional document identifier (defaults to doc_001)",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Parsed document sections",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "parsed_doc": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "document_id": {"type": "string"},
                                                        "parsed_header_text": {"type": "string"},
                                                        "parsed_text": {"type": "string"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Missing required parameters"},
                        "500": {"description": "LLM parsing failed"},
                    },
                }
            },
        },
    }


@app.get("/openapi.json")
def openapi():
    return jsonify(build_openapi_spec())


@app.get("/docs")
def docs():
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Web Gather API Docs</title>
        <link
          rel="stylesheet"
          href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
        />
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
          window.onload = () => {
            SwaggerUIBundle({
              url: "/openapi.json",
              dom_id: "#swagger-ui"
            });
          };
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


def utc_now():
    return datetime.now(timezone.utc)


def get_model(model_name: str) -> SentenceTransformer:
    model = _model_cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    return model


def normalize_results(raw_result):
    # Handle Firecrawl CrawlJob response (has .data attribute)
    if hasattr(raw_result, "data") and isinstance(raw_result.data, list):
        return raw_result.data
    # Handle Firecrawl SearchData response (has .web attribute)
    if hasattr(raw_result, "web") and raw_result.web is not None:
        return raw_result.web
    # Handle dict responses (legacy compatibility)
    if isinstance(raw_result, dict):
        if "data" in raw_result:
            return raw_result["data"]
        if "results" in raw_result:
            return raw_result["results"]
        if "pages" in raw_result:
            return raw_result["pages"]
    if isinstance(raw_result, list):
        return raw_result
    return []


def get_page_attr(page, attr, default=""):
    """Get attribute from page object or dict."""
    if hasattr(page, attr):
        return getattr(page, attr, default) or default
    if isinstance(page, dict):
        return page.get(attr, default) or default
    return default


def combine_pages(pages):
    combined = []
    for page in pages:
        # Handle Firecrawl Document objects (url/title in metadata)
        if hasattr(page, "metadata") and page.metadata:
            url = getattr(page.metadata, "url", "") or ""
            title = getattr(page.metadata, "title", "") or ""
        else:
            url = get_page_attr(page, "url")
            title = get_page_attr(page, "title")

        # Get content - try markdown first (Firecrawl), then fallbacks
        content = (
            get_page_attr(page, "markdown")
            or get_page_attr(page, "raw_content")
            or get_page_attr(page, "content")
            or get_page_attr(page, "text")
        )
        if not content:
            continue
        header = "Source"
        if title:
            header = f"{title} | Source"
        combined.append(f"{header}: {url}\n{content}".strip())
    return "\n\n".join(combined).strip()


def chunk_text(text, chunk_size=1200, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(query_embedding, chunk_embeddings):
    if not chunk_embeddings:
        return np.array([])
    chunk_matrix = np.vstack(chunk_embeddings)
    return chunk_matrix @ query_embedding


def get_embedding_model_name(database_name: str):
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


def index_document_chunks(document_id: str, text: str, database_name: str, collection_name: str):
    """
    Create vector-indexed chunks for a document.
    Returns dict with indexing results or None if embedding model not configured.
    """
    logger.info("Starting indexing for document: %s", document_id)

    model_name = get_embedding_model_name(database_name)
    if not model_name:
        logger.info("Skipping indexing - no embedding model configured for database: %s", database_name)
        return None

    logger.info("Chunking document text (length: %d characters)", len(text))
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("Document %s has no content to chunk", document_id)
        return None

    logger.info("Created %d chunks from document %s", len(chunks), document_id)

    logger.info("Loading embedding model: %s", model_name)
    model = get_model(model_name)

    logger.info("Generating embeddings for %d chunks", len(chunks))
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("Embeddings generated successfully (dimension: %d)", embeddings.shape[1])

    db = mongo_client[database_name]
    chunk_collection = f"{collection_name}_chunks"

    logger.info("Clearing existing chunks for document %s in collection %s", document_id, chunk_collection)
    delete_result = db[chunk_collection].delete_many({"document_id": document_id})
    logger.info("Deleted %d existing chunks", delete_result.deleted_count)

    chunk_docs = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_docs.append(
            {
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding.tolist(),
                "created_at": utc_now(),
            }
        )

    logger.info("Inserting %d chunks into collection %s", len(chunk_docs), chunk_collection)
    db[chunk_collection].insert_many(chunk_docs)

    wg_db = mongo_client[WEB_GATHER_DB]
    wg_db[DOCUMENTS_COLLECTION].update_one(
        {"document_id": document_id},
        {
            "$set": {
                "indexed_at": utc_now(),
                "chunk_count": len(chunk_docs),
                "embedding_model": model_name,
                "chunk_collection": chunk_collection,
            }
        },
    )

    logger.info("Indexing complete for document %s: %d chunks indexed with model %s",
                document_id, len(chunk_docs), model_name)

    return {
        "chunks_indexed": len(chunk_docs),
        "embedding_model": model_name,
        "chunk_collection": chunk_collection,
    }


def serialize_search_result(result):
    """Convert Firecrawl search result object to dict."""
    if isinstance(result, dict):
        return result
    # Handle SearchResultWeb or Document objects
    data = {}
    if hasattr(result, "url"):
        data["url"] = result.url
    if hasattr(result, "title"):
        data["title"] = result.title
    if hasattr(result, "description"):
        data["description"] = result.description
    if hasattr(result, "markdown"):
        data["markdown"] = result.markdown
    if hasattr(result, "metadata") and result.metadata:
        data["url"] = getattr(result.metadata, "url", None) or data.get("url")
        data["title"] = getattr(result.metadata, "title", None) or data.get("title")
    return data


@app.post("/gather")
def gather():
    logger.info("POST /gather - Starting web search")
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not query:
        logger.warning("POST /gather - Missing required parameter: query")
        return jsonify({"error": "query is required"}), 400

    logger.info("POST /gather - Searching for: %s", query)
    results = firecrawl_client.search(
        query=query,
        limit=10,
    )
    normalized = normalize_results(results)
    serialized = [serialize_search_result(r) for r in normalized]
    logger.info("POST /gather - Found %d results for query: %s", len(serialized), query)
    return jsonify({"query": query, "results": serialized})


@app.post("/ingest")
def ingest():
    logger.info("POST /ingest - Starting document ingestion")
    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    depth = int(payload.get("depth", 1))
    breadth = int(payload.get("breadth", 5))
    database_name = payload.get("database")
    collection_name = payload.get("collection")

    if not url or not database_name or not collection_name:
        logger.warning("POST /ingest - Missing required parameters")
        return (
            jsonify(
                {
                    "error": "url, database, and collection are required",
                }
            ),
            400,
        )

    logger.info("POST /ingest - Crawling URL: %s (depth=%d, breadth=%d)", url, depth, breadth)
    try:
        crawl_result = firecrawl_client.crawl(
            url=url,
            max_discovery_depth=depth,
            limit=breadth,
        )
    except Exception as exc:
        logger.error("POST /ingest - Crawl failed for URL %s: %s", url, exc)
        return jsonify({"error": f"crawl failed: {exc}"}), 500

    pages = normalize_results(crawl_result)
    logger.info("POST /ingest - Crawl returned %d pages", len(pages))

    combined_text = combine_pages(pages)
    if not combined_text:
        logger.warning("POST /ingest - Crawl returned no content for URL: %s", url)
        return jsonify({"error": "crawl returned no content"}), 400

    logger.info("POST /ingest - Combined text length: %d characters", len(combined_text))

    document_id = str(uuid.uuid4())
    logger.info("POST /ingest - Created document ID: %s", document_id)

    db = mongo_client[database_name]
    logger.info("POST /ingest - Storing document in %s.%s", database_name, collection_name)
    db[collection_name].insert_one(
        {
            "_id": document_id,
            "source_url": url,
            "text": combined_text,
            "created_at": utc_now(),
        }
    )

    description = combined_text[:300]
    wg_db = mongo_client[WEB_GATHER_DB]
    logger.info("POST /ingest - Creating document metadata record")
    wg_db[DOCUMENTS_COLLECTION].insert_one(
        {
            "document_id": document_id,
            "database_name": database_name,
            "collection_name": collection_name,
            "source_url": url,
            "description": description,
            "created_at": utc_now(),
        }
    )

    chunk_collection = f"{collection_name}_chunks"
    wg_db[DOCUMENTS_COLLECTION].update_one(
        {"document_id": document_id},
        {"$set": {"chunk_collection": chunk_collection}},
    )
    logger.info("POST /ingest - Creating index on chunk collection: %s", chunk_collection)
    db[chunk_collection].create_index("document_id")

    # Automatically create vector-indexed chunks if embedding model is configured
    logger.info("POST /ingest - Attempting to create vector-indexed chunks")
    index_result = index_document_chunks(
        document_id=document_id,
        text=combined_text,
        database_name=database_name,
        collection_name=collection_name,
    )

    response_data = {
        "document_id": document_id,
        "pages": len(pages),
        "database_name": database_name,
        "collection_name": collection_name,
        "chunk_collection": chunk_collection,
    }

    if index_result:
        response_data["indexed"] = True
        response_data["chunks_indexed"] = index_result["chunks_indexed"]
        response_data["embedding_model"] = index_result["embedding_model"]
        logger.info("POST /ingest - Document ingested and indexed successfully: %s", document_id)
    else:
        response_data["indexed"] = False
        response_data["message"] = "Document stored but not indexed - configure embedding model first"
        logger.info("POST /ingest - Document ingested but not indexed (no embedding model): %s", document_id)

    return jsonify(response_data)


@app.get("/documents")
def list_documents():
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


@app.get("/collections")
def list_collections():
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


@app.post("/index")
def index_document():
    logger.info("POST /index - Starting document indexing")
    payload = request.get_json(silent=True) or {}
    document_id = payload.get("document_id")
    if not document_id:
        logger.warning("POST /index - Missing required parameter: document_id")
        return jsonify({"error": "document_id is required"}), 400

    logger.info("POST /index - Looking up document: %s", document_id)
    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        logger.warning("POST /index - Document not found: %s", document_id)
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]
    logger.info("POST /index - Document found in %s.%s", database_name, collection_name)

    # Check if embedding model is configured
    model_name = get_embedding_model_name(database_name)
    if not model_name:
        logger.error("POST /index - No embedding model configured for database: %s", database_name)
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[database_name]
    document = db[collection_name].find_one({"_id": document_id})
    if not document:
        logger.warning("POST /index - Document content not found: %s", document_id)
        return jsonify({"error": "document not found"}), 404

    text = document.get("text", "")
    if not text:
        logger.warning("POST /index - Document has no text content: %s", document_id)
        return jsonify({"error": "document has no content"}), 400

    # Use shared indexing function
    index_result = index_document_chunks(
        document_id=document_id,
        text=text,
        database_name=database_name,
        collection_name=collection_name,
    )

    if not index_result:
        logger.error("POST /index - Indexing failed for document: %s", document_id)
        return jsonify({"error": "indexing failed"}), 500

    logger.info("POST /index - Successfully indexed document: %s", document_id)
    return jsonify(
        {
            "document_id": document_id,
            "chunks_indexed": index_result["chunks_indexed"],
            "embedding_model": index_result["embedding_model"],
            "chunk_collection": index_result["chunk_collection"],
        }
    )


@app.get("/search")
def search():
    logger.info("GET /search - Starting vector search")
    document_id = request.args.get("document_id")
    query = request.args.get("query")
    if not document_id or not query:
        logger.warning("GET /search - Missing required parameters")
        return jsonify({"error": "document_id and query are required"}), 400

    logger.info("GET /search - Searching document %s for: %s", document_id, query)
    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        logger.warning("GET /search - Document not found: %s", document_id)
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]
    model_name = get_embedding_model_name(database_name)
    if not model_name:
        logger.error("GET /search - No embedding model configured for database: %s", database_name)
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[database_name]
    chunk_collection = f"{collection_name}_chunks"
    logger.info("GET /search - Loading chunks from %s", chunk_collection)
    chunks = list(
        db[chunk_collection].find({"document_id": document_id}, {"_id": 0})
    )
    if not chunks:
        logger.warning("GET /search - No chunks indexed for document: %s", document_id)
        return jsonify({"error": "no chunks indexed for document"}), 400

    logger.info("GET /search - Found %d chunks, generating query embedding", len(chunks))
    model = get_model(model_name)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    chunk_embeddings = [np.array(c["embedding"]) for c in chunks]

    logger.info("GET /search - Computing cosine similarity scores")
    scores = cosine_similarity(query_embedding, chunk_embeddings)

    results = []
    for chunk, score in zip(chunks, scores):
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "percent_match": max(0.0, min(100.0, float(score) * 100.0)),
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)

    top_results = results[:5]
    logger.info("GET /search - Returning top %d results (best score: %.3f)",
                len(top_results), top_results[0]["score"] if top_results else 0)
    return jsonify(
        {
            "document_id": document_id,
            "query": query,
            "results": top_results,
        }
    )


@app.get("/embedding-models")
def list_embedding_models():
    logger.info("GET /embedding-models - Listing embedding models")
    database_name = request.args.get("database_name")
    wg_db = mongo_client[WEB_GATHER_DB]
    query = {"database_name": database_name} if database_name else {}
    models = list(wg_db[EMBEDDING_MODEL_COLLECTION].find(query, {"_id": 0}))
    logger.info("GET /embedding-models - Found %d models", len(models))
    return jsonify({"models": models})


@app.post("/embedding-models")
def add_embedding_model():
    logger.info("POST /embedding-models - Adding embedding model")
    payload = request.get_json(silent=True) or {}
    database_name = payload.get("database_name")
    model_name = payload.get("model_name")
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


@app.get("/parse_llm")
def parse_llm():
    logger.info("GET /parse_llm - Starting LLM document parsing")
    document = request.args.get("document")
    parsing_prompt = request.args.get("parsing_prompt")
    document_id = request.args.get("document_id", "doc_001")

    if not document or not parsing_prompt:
        logger.warning("GET /parse_llm - Missing required parameters")
        return jsonify({"error": "document and parsing_prompt are required"}), 400

    logger.info("GET /parse_llm - Parsing document (length: %d) with prompt: %s...",
                len(document), parsing_prompt[:50])

    system_prompt = """You are a document parsing assistant. Your task is to parse documents according to specific instructions and return structured JSON output.

You must return ONLY valid JSON with the following structure:
- A top-level object with a "parsed_doc" key
- The "parsed_doc" value should be an array of objects
- Each object in the array represents one parsed section and must contain:
  - "document_id": A string identifier for the document (use the provided document_id)
  - "parsed_header_text": A string containing the header, title, or identifying text for this section
  - "parsed_text": A string containing the main content/body text for this section

Important guidelines:
- Preserve the original text accurately - do not paraphrase or summarize unless explicitly instructed
- Each section should be complete and meaningful on its own
- Escape special characters properly in the JSON (quotes, newlines, etc.)
- The parsed_header_text should be concise but descriptive enough to identify what the section contains
- The parsed_text should contain the substantive content of that section
- Return ONLY the JSON object, no additional text or markdown formatting"""

    user_message = f"""Parse the following document according to the parsing instructions provided.

<document>
{document}
</document>

<parsing_prompt>
{parsing_prompt}
</parsing_prompt>

<document_id>{document_id}</document_id>

Return ONLY the JSON output with the parsed document sections."""

    try:
        logger.info("GET /parse_llm - Calling Anthropic API (model: claude-3-haiku-20240307)")
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=system_prompt,
        )

        response_text = message.content[0].text.strip()
        logger.info("GET /parse_llm - Received response (length: %d)", len(response_text))

        # Try to extract JSON if wrapped in code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (code block markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        parsed_result = json.loads(response_text)
        section_count = len(parsed_result.get("parsed_doc", []))
        logger.info("GET /parse_llm - Successfully parsed document into %d sections", section_count)
        return jsonify(parsed_result)

    except json.JSONDecodeError as e:
        logger.error("GET /parse_llm - Failed to parse LLM response as JSON: %s", e)
        return jsonify({"error": f"Failed to parse LLM response as JSON: {str(e)}"}), 500
    except Exception as e:
        logger.error("GET /parse_llm - LLM parsing failed: %s", e)
        return jsonify({"error": f"LLM parsing failed: {str(e)}"}), 500


if __name__ == "__main__":
    logger.info("Starting Web Gather API server on port 5000")
    app.run(debug=True, port=5000)
