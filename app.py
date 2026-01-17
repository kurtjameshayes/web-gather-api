import os
import uuid
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")

WEB_GATHER_DB = "web-gather"
DOCUMENTS_COLLECTION = "documents"
EMBEDDING_MODEL_COLLECTION = "embedding_model"

app = Flask(__name__)
mongo_client = MongoClient(MONGODB_URI)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

_model_cache = {}


def utc_now():
    return datetime.now(timezone.utc)


def get_model(model_name: str) -> SentenceTransformer:
    model = _model_cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    return model


def normalize_results(raw_result):
    if isinstance(raw_result, dict):
        if "results" in raw_result:
            return raw_result["results"]
        if "pages" in raw_result:
            return raw_result["pages"]
    if isinstance(raw_result, list):
        return raw_result
    return []


def combine_pages(pages):
    combined = []
    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        content = page.get("raw_content") or page.get("content") or page.get("text") or ""
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
    wg_db = mongo_client[WEB_GATHER_DB]
    record = wg_db[EMBEDDING_MODEL_COLLECTION].find_one(
        {"database_name": database_name}
    )
    if not record:
        return None
    return record.get("model_name")


@app.post("/gather")
def gather():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not query:
        return jsonify({"error": "query is required"}), 400

    results = tavily_client.search(
        query=query,
        max_results=10,
        include_raw_content=True,
    )
    return jsonify({"query": query, "results": normalize_results(results)})


@app.post("/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    depth = int(payload.get("depth", 1))
    breadth = int(payload.get("breadth", 5))
    database_name = payload.get("database")
    collection_name = payload.get("collection")

    if not url or not database_name or not collection_name:
        return (
            jsonify(
                {
                    "error": "url, database, and collection are required",
                }
            ),
            400,
        )

    try:
        crawl_result = tavily_client.crawl(
            url=url,
            max_depth=depth,
            max_breadth=breadth,
        )
    except Exception as exc:
        return jsonify({"error": f"crawl failed: {exc}"}), 500

    pages = normalize_results(crawl_result)
    combined_text = combine_pages(pages)
    if not combined_text:
        return jsonify({"error": "crawl returned no content"}), 400

    document_id = str(uuid.uuid4())
    db = mongo_client[database_name]
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
    db[chunk_collection].create_index("document_id")

    return jsonify(
        {
            "document_id": document_id,
            "pages": len(pages),
            "database_name": database_name,
            "collection_name": collection_name,
            "chunk_collection": chunk_collection,
        }
    )


@app.get("/documents")
def list_documents():
    database_name = request.args.get("database_name")
    collection_name = request.args.get("collection_name")
    if not database_name or not collection_name:
        return jsonify({"error": "database_name and collection_name are required"}), 400

    wg_db = mongo_client[WEB_GATHER_DB]
    docs = list(
        wg_db[DOCUMENTS_COLLECTION].find(
            {"database_name": database_name, "collection_name": collection_name},
            {"_id": 0},
        )
    )
    return jsonify({"documents": docs})


@app.get("/collections")
def list_collections():
    database_name = request.args.get("database_name")
    if not database_name:
        return jsonify({"error": "database_name is required"}), 400

    wg_db = mongo_client[WEB_GATHER_DB]
    collections = wg_db[DOCUMENTS_COLLECTION].distinct(
        "collection_name", {"database_name": database_name}
    )
    return jsonify({"database_name": database_name, "collections": collections})


@app.post("/index")
def index_document():
    payload = request.get_json(silent=True) or {}
    document_id = payload.get("document_id")
    if not document_id:
        return jsonify({"error": "document_id is required"}), 400

    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]
    model_name = get_embedding_model_name(database_name)
    if not model_name:
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[database_name]
    document = db[collection_name].find_one({"_id": document_id})
    if not document:
        return jsonify({"error": "document not found"}), 404

    chunks = chunk_text(document.get("text", ""))
    if not chunks:
        return jsonify({"error": "document has no content"}), 400

    model = get_model(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    chunk_collection = f"{collection_name}_chunks"
    db[chunk_collection].delete_many({"document_id": document_id})
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
    db[chunk_collection].insert_many(chunk_docs)

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

    return jsonify(
        {
            "document_id": document_id,
            "chunks_indexed": len(chunk_docs),
            "embedding_model": model_name,
            "chunk_collection": chunk_collection,
        }
    )


@app.post("/search")
def search():
    payload = request.get_json(silent=True) or {}
    document_id = payload.get("document_id")
    query = payload.get("query")
    if not document_id or not query:
        return jsonify({"error": "document_id and query are required"}), 400

    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]
    model_name = get_embedding_model_name(database_name)
    if not model_name:
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[database_name]
    chunk_collection = f"{collection_name}_chunks"
    chunks = list(
        db[chunk_collection].find({"document_id": document_id}, {"_id": 0})
    )
    if not chunks:
        return jsonify({"error": "no chunks indexed for document"}), 400

    model = get_model(model_name)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    chunk_embeddings = [np.array(c["embedding"]) for c in chunks]
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

    return jsonify(
        {
            "document_id": document_id,
            "query": query,
            "results": results[:5],
        }
    )


@app.get("/embedding-models")
def list_embedding_models():
    database_name = request.args.get("database_name")
    wg_db = mongo_client[WEB_GATHER_DB]
    query = {"database_name": database_name} if database_name else {}
    models = list(wg_db[EMBEDDING_MODEL_COLLECTION].find(query, {"_id": 0}))
    return jsonify({"models": models})


@app.post("/embedding-models")
def add_embedding_model():
    payload = request.get_json(silent=True) or {}
    database_name = payload.get("database_name")
    model_name = payload.get("model_name")
    if not database_name or not model_name:
        return jsonify({"error": "database_name and model_name are required"}), 400

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

    return jsonify({"database_name": database_name, "model_name": model_name})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
