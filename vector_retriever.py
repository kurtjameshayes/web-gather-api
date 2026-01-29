"""MongoDB vector search wrapper for statute retrieval."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cache import SimpleLRUCache
from compliance_config import ComplianceConfig
from compliance_utils import hash_text, safe_truncate
from embedder import Embedder


@dataclass
class StatuteCandidate:
    statute_id: str
    jurisdiction: str
    title: str
    section_id: str
    chunk_text: str
    score: float
    chunk_id: str


class VectorRetriever:
    def __init__(
        self,
        mongo_client: Any,
        embedder: Embedder,
        config: ComplianceConfig,
        cache: SimpleLRUCache,
    ) -> None:
        self._mongo_client = mongo_client
        self._embedder = embedder
        self._config = config
        self._cache = cache

    async def retrieve(
        self,
        database: str,
        section_text: str,
        jurisdiction: str,
        statute_corpus_id: Optional[str],
        top_k: int,
    ) -> List[StatuteCandidate]:
        cache_key = hash_text(
            f"{database}:{jurisdiction}:{statute_corpus_id or ''}:{top_k}:{section_text}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        query_vector = await self._embedder.embed(section_text)
        if not query_vector:
            return []

        if self._config.use_embeddings_collection:
            results = await self._retrieve_from_embeddings(
                database, query_vector, statute_corpus_id, top_k
            )
        else:
            results = await self._retrieve_from_statutes(
                database, query_vector, jurisdiction, statute_corpus_id, top_k
            )

        self._cache.set(cache_key, results)
        return results

    async def _retrieve_from_statutes(
        self,
        database: str,
        query_vector: List[float],
        jurisdiction: str,
        statute_corpus_id: Optional[str],
        top_k: int,
    ) -> List[StatuteCandidate]:
        filter_doc: Dict[str, Any] = {
            self._config.statute_jurisdiction_field: jurisdiction,
        }
        if statute_corpus_id:
            filter_doc["$or"] = [
                {self._config.statute_corpus_field: statute_corpus_id},
                {f"metadata.{self._config.statute_corpus_field}": statute_corpus_id},
            ]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._config.vector_index_name,
                    "path": self._config.vector_field,
                    "queryVector": query_vector,
                    "numCandidates": max(top_k * 10, top_k),
                    "limit": top_k,
                    "filter": filter_doc,
                }
            },
            {
                "$project": {
                    self._config.statute_id_field: 1,
                    self._config.statute_title_field: 1,
                    self._config.statute_text_field: 1,
                    self._config.statute_section_id_field: 1,
                    self._config.statute_jurisdiction_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        collection = self._mongo_client[database][self._config.statutes_collection]

        def run_aggregate() -> List[Dict[str, Any]]:
            return list(collection.aggregate(pipeline))

        raw_results = await asyncio.to_thread(run_aggregate)
        candidates = []
        for result in raw_results:
            candidates.append(
                StatuteCandidate(
                    statute_id=str(result.get(self._config.statute_id_field, "")),
                    jurisdiction=str(result.get(self._config.statute_jurisdiction_field, "")),
                    title=str(result.get(self._config.statute_title_field, "")),
                    section_id=str(result.get(self._config.statute_section_id_field, "")),
                    chunk_text=safe_truncate(
                        str(result.get(self._config.statute_text_field, "")),
                        self._config.max_statute_chunk_chars,
                    ),
                    score=float(result.get("score", 0.0)),
                    chunk_id=str(result.get(self._config.statute_section_id_field, "0")),
                )
            )
        return candidates

    async def _retrieve_from_embeddings(
        self,
        database: str,
        query_vector: List[float],
        statute_corpus_id: Optional[str],
        top_k: int,
    ) -> List[StatuteCandidate]:
        filter_doc: Dict[str, Any] = {}
        collection_tag = statute_corpus_id or self._config.statute_collection_tag
        if collection_tag:
            filter_doc[self._config.embedding_collection_tag_field] = collection_tag

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._config.vector_index_name,
                    "path": self._config.embedding_vector_field,
                    "queryVector": query_vector,
                    "numCandidates": max(top_k * 10, top_k),
                    "limit": top_k,
                    "filter": filter_doc,
                }
            },
            {
                "$project": {
                    self._config.embedding_doc_id_field: 1,
                    self._config.embedding_text_field: 1,
                    self._config.embedding_chunk_id_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        embedding_collection = self._mongo_client[database][self._config.embeddings_collection]

        def run_aggregate() -> List[Dict[str, Any]]:
            return list(embedding_collection.aggregate(pipeline))

        raw_results = await asyncio.to_thread(run_aggregate)
        doc_ids = [r.get(self._config.embedding_doc_id_field) for r in raw_results if r.get(self._config.embedding_doc_id_field)]

        statutes_collection = self._mongo_client[database][self._config.statutes_collection]

        def run_fetch() -> List[Dict[str, Any]]:
            return list(statutes_collection.find({self._config.statute_id_field: {"$in": doc_ids}}))

        statute_docs = await asyncio.to_thread(run_fetch) if doc_ids else []
        statute_map = {
            str(doc.get(self._config.statute_id_field)): doc for doc in statute_docs
        }

        candidates = []
        for result in raw_results:
            doc_id = str(result.get(self._config.embedding_doc_id_field, ""))
            statute_doc = statute_map.get(doc_id, {})
            candidates.append(
                StatuteCandidate(
                    statute_id=doc_id,
                    jurisdiction=str(statute_doc.get(self._config.statute_jurisdiction_field, "")),
                    title=str(statute_doc.get(self._config.statute_title_field, "")),
                    section_id=str(statute_doc.get(self._config.statute_section_id_field, "")),
                    chunk_text=safe_truncate(
                        str(result.get(self._config.embedding_text_field, "")),
                        self._config.max_statute_chunk_chars,
                    ),
                    score=float(result.get("score", 0.0)),
                    chunk_id=str(result.get(self._config.embedding_chunk_id_field, "0")),
                )
            )
        return candidates
