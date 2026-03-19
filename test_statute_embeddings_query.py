"""Simple script to run a vector search on statute_embeddings and print results.

Requires: MONGODB_URI. Embedding model is read from web-gather.embedding_model
(database_name = compliance_database, e.g. privacy-compliance); fallback to config if missing.
Uses compliance_database and embeddings_collection from config (e.g. statute_embeddings).

Env:
  SKIP_JURISDICTION_FILTER=1  Omit jurisdiction filter (diagnose index/vector vs filter).
  STATUTE_EMBEDDINGS_JURISDICTION  Jurisdiction value for filter (default: California).

Run: python test_statute_embeddings_query.py [--no-filter]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import SimpleLRUCache
from compliance_config import load_config
from db import get_embedding_model_name, init_db
from embedder import Embedder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector search on statute_embeddings")
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Omit jurisdiction filter (same as SKIP_JURISDICTION_FILTER=1)",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    skip_filter = args.no_filter or (os.getenv("SKIP_JURISDICTION_FILTER", "").strip().lower() in ("1", "true", "yes"))

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("MONGODB_URI not set")
        return

    config = load_config()
    database = config.compliance_database
    collection_name = config.embeddings_collection
    index_name = config.vector_index_name
    vector_path = config.embedding_vector_field
    jurisdiction_field = config.statute_jurisdiction_field or "jurisdiction"
    text_field = config.embedding_text_field
    doc_id_field = config.embedding_doc_id_field
    chunk_id_field = config.embedding_chunk_id_field

    jurisdiction_filter = os.getenv("STATUTE_EMBEDDINGS_JURISDICTION", "California").strip() or "California"

    query_text = "personal data privacy rights"
    top_k = 3

    print(f"Connecting to MongoDB...")
    client = MongoClient(mongodb_uri)
    init_db(client)

    # Embedding model from web-gather.embedding_model (database_name = compliance_database)
    model_name = get_embedding_model_name(config.compliance_database)
    if not model_name:
        model_name = config.embedding_model_name
        print(f"No embedding model for {config.compliance_database!r} in DB; using config: {model_name!r}")
    else:
        print(f"Using embedding model from web-gather.embedding_model: {model_name!r}")

    coll = client[database][collection_name]

    # Diagnostic: show one document shape (excluding vector blobs)
    sample = coll.find_one({}, {"embedding": 0, "vector": 0})
    vec_len = None
    if sample:
        keys = list(sample.keys())
        jur_val = sample.get(jurisdiction_field) or sample.get("jurisdiction")
        vec_doc = coll.find_one({"$or": [{"embedding": {"$exists": True}}, {"vector": {"$exists": True}}]}, {"embedding": 1, "vector": 1})
        has_embedding = vec_doc and isinstance(vec_doc.get("embedding"), (list, tuple))
        has_vector = vec_doc and isinstance(vec_doc.get("vector"), (list, tuple))
        vec_len = None
        if vec_doc:
            vec = vec_doc.get("embedding") or vec_doc.get("vector")
            vec_len = len(vec) if isinstance(vec, (list, tuple)) else None
        print(f"Sample doc keys: {keys}, {jurisdiction_field}={jur_val!r}, has_embedding={has_embedding}, has_vector={has_vector}, vector_dim={vec_len}")
    else:
        print("No documents in collection.")

    print(f"Embedding query: {query_text!r}")
    cache = SimpleLRUCache(config.cache_size, config.cache_ttl_seconds)
    embedder = Embedder(model_name, cache)
    query_vector = await embedder.embed(query_text)
    if not query_vector:
        print("Embedding failed")
        return

    print(f"Query vector dim: {len(query_vector)}")
    # Dimension mismatch causes 0 results: index must match query vector length
    if sample and vec_len is not None and len(query_vector) != vec_len:
        print(f"WARNING: Query vector dim={len(query_vector)} but collection vectors have dim={vec_len}. Index will not match. Re-index with same model or use a {vec_len}-dim model.")

    if skip_filter:
        filter_doc = {}
    else:
        filter_doc = {jurisdiction_field: jurisdiction_filter}

    # Log filter and index/path for debugging
    print(f"index: {index_name!r}, path: {vector_path!r}, filter: {filter_doc}")

    vector_search_stage = {
        "index": index_name,
        "path": vector_path,
        "queryVector": query_vector,
        "numCandidates": max(top_k * 10, top_k),
        "limit": top_k,
    }
    if filter_doc:
        vector_search_stage["filter"] = filter_doc

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {
            "$project": {
                doc_id_field: 1,
                text_field: 1,
                chunk_id_field: 1,
                jurisdiction_field: 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    filter_desc = "no filter" if skip_filter else f"jurisdiction={jurisdiction_filter!r}"
    print(f"Vector search on {database}.{collection_name} (index={index_name!r}, {filter_desc})...")
    try:
        results = list(coll.aggregate(pipeline))
    except Exception as e:
        print(f"ERROR: Vector search failed: {e}")
        client.close()
        return

    print(f"\nFound {len(results)} result(s):\n")
    if len(results) == 0 and sample and vec_len is not None and len(query_vector) == vec_len:
        print("Hint: Dimensions match. If index exists, check Atlas index name/path and that the index includes this collection and filter.")
        print()
    for i, doc in enumerate(results, 1):
        score = doc.get("score")
        doc_id = doc.get(doc_id_field)
        chunk_id = doc.get(chunk_id_field)
        jur = doc.get(jurisdiction_field)
        text = doc.get(text_field, "")
        snippet = (text[:200] + "...") if len(text) > 200 else text
        print(f"--- Result {i} ---")
        print(f"  score:      {score}")
        print(f"  doc_id:     {doc_id}")
        print(f"  chunk_id:   {chunk_id}")
        print(f"  jurisdiction: {jur}")
        print(f"  text:       {snippet}")
        print()

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
