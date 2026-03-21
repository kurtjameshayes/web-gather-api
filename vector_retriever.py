"""MongoDB vector search wrapper for statute retrieval."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from async_utils import run_in_thread

# Jurisdiction for vector search (override all callers). Configurable via env for DB values like "CA".
VECTOR_SEARCH_JURISDICTION = (os.getenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION") or "California").strip() or "California"

from cache import SimpleLRUCache
from compliance_config import ComplianceConfig
from compliance_utils import hash_text, jurisdiction_filter_values, normalize_jurisdiction, safe_truncate
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
    chunk_header_text: str = ""


@dataclass
class SubchunkPair:
    """Statute subchunk doc and best-matching policy subchunk doc for gap analysis."""

    statute_doc: Dict[str, Any]
    policy_doc: Dict[str, Any]
    score: float


@dataclass
class RetrievePolicySubchunksResult:
    """Result of retrieve_policy_subchunks_for_statute_subchunks with metadata."""

    pairs: List[SubchunkPair]
    statute_subchunks_considered: int


@dataclass
class ChunkPair:
    """Statute chunk doc and best-matching policy chunk doc for chunk-level gap analysis (v2)."""

    statute_doc: Dict[str, Any]
    policy_doc: Dict[str, Any]
    score: float


@dataclass
class RetrievePolicyChunksResult:
    """Result of retrieve_policy_chunks_for_statute_chunks with metadata."""

    pairs: List[ChunkPair]
    statute_chunks_considered: int


@dataclass
class PolicyMatch:
    """A single policy subchunk match from vector search (v3 v1 design)."""

    text: str
    score: float
    section_id: Optional[str] = None
    parent_context: Optional[str] = None  # Enclosing chunk from policy_embeddings


@dataclass
class StatutePolicyPairV3:
    """Statute subchunk and its top-k policy subchunk matches (v3 v1 design)."""

    statute_doc: Dict[str, Any]
    policy_matches: List[PolicyMatch]
    above_threshold_count: int  # Matches with score >= threshold
    statute_parent_context: Optional[str] = None  # Enclosing chunk from statute_embeddings


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
                database, query_vector, jurisdiction, statute_corpus_id, top_k
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
        # Use passed jurisdiction; resolvable via jurisdiction_filter_values (e.g. CA -> California).
        # Env override for backward compat when DB uses non-standard values.
        eff_jurisdiction = (jurisdiction or "").strip() or VECTOR_SEARCH_JURISDICTION
        filter_values = jurisdiction_filter_values(normalize_jurisdiction(eff_jurisdiction))
        if not filter_values:
            filter_values = [VECTOR_SEARCH_JURISDICTION]
        filter_doc: Dict[str, Any] = {
            self._config.statute_jurisdiction_field: (
                {"$in": filter_values} if len(filter_values) > 1 else filter_values[0]
            ),
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
                    self._config.statute_section_field: 1,
                    self._config.statute_section_id_field: 1,
                    self._config.statute_jurisdiction_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        header_field = getattr(self._config, "statute_chunk_header_field", None)
        if header_field:
            pipeline[1]["$project"][header_field] = 1

        collection = self._mongo_client[database][self._config.statutes_collection]

        def run_aggregate() -> List[Dict[str, Any]]:
            return list(collection.aggregate(pipeline))

        raw_results = await run_in_thread(run_aggregate)
        candidates = []
        sec_field = self._config.statute_section_field
        sec_id_field = self._config.statute_section_id_field
        for result in raw_results:
            section_val = result.get(sec_field) or result.get(sec_id_field)
            candidates.append(
                StatuteCandidate(
                    statute_id=str(result.get(self._config.statute_id_field, "")),
                    jurisdiction=str(result.get(self._config.statute_jurisdiction_field, "")),
                    title=str(result.get(self._config.statute_title_field, "")),
                    section_id=str(section_val or ""),
                    chunk_text=safe_truncate(
                        str(result.get(self._config.statute_text_field, "")),
                        self._config.max_statute_chunk_chars,
                    ),
                    score=float(result.get("score", 0.0)),
                    chunk_id=str(result.get(self._config.statute_section_id_field, "0")),
                    chunk_header_text=str(
                        result.get(getattr(self._config, "statute_chunk_header_field", None) or "", "")
                    ),
                )
            )
        return candidates

    async def retrieve_by_queries(
        self,
        database: str,
        jurisdiction: str,
        queries: List[str],
        top_k_per_query: int,
        statute_corpus_id: Optional[str] = None,
    ) -> List[StatuteCandidate]:
        """Retrieve statute chunks for multiple semantic queries in one jurisdiction, deduplicated by chunk id."""
        seen: Dict[str, StatuteCandidate] = {}
        for query in queries:
            candidates = await self.retrieve(
                database=database,
                section_text=query,
                jurisdiction=jurisdiction,
                statute_corpus_id=statute_corpus_id,
                top_k=top_k_per_query,
            )
            for c in candidates:
                key = f"{c.statute_id}:{c.chunk_id}"
                if key not in seen or c.score > seen[key].score:
                    seen[key] = c
        result = list(seen.values())
        result.sort(key=lambda x: -x.score)
        return result

    async def _retrieve_from_embeddings(
        self,
        database: str,
        query_vector: List[float],
        jurisdiction: str,
        statute_corpus_id: Optional[str],
        top_k: int,
    ) -> List[StatuteCandidate]:
        filter_doc: Dict[str, Any] = {}
        collection_tag = statute_corpus_id or self._config.statute_collection_tag
        if collection_tag:
            filter_doc[self._config.embedding_collection_tag_field] = collection_tag
        # Use passed jurisdiction; resolvable via jurisdiction_filter_values (e.g. CA -> California).
        eff_jurisdiction = (jurisdiction or "").strip() or VECTOR_SEARCH_JURISDICTION
        filter_values = jurisdiction_filter_values(normalize_jurisdiction(eff_jurisdiction))
        if not filter_values:
            filter_values = [VECTOR_SEARCH_JURISDICTION]
        field = self._config.statute_jurisdiction_field or "jurisdiction"
        filter_doc[field] = (
            {"$in": filter_values} if len(filter_values) > 1 else filter_values[0]
        )

        vector_search_stage: Dict[str, Any] = {
            "index": self._config.vector_index_name,
            "path": self._config.embedding_vector_field,
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
                    self._config.embedding_doc_id_field: 1,
                    self._config.embedding_text_field: 1,
                    self._config.embedding_chunk_id_field: 1,
                    self._config.statute_jurisdiction_field: 1,
                    self._config.statute_section_field: 1,
                    self._config.statute_section_id_field: 1,
                    self._config.statute_chunk_header_field: 1,
                    "jurisdiction": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        embedding_collection = self._mongo_client[database][self._config.embeddings_collection]

        def run_aggregate() -> List[Dict[str, Any]]:
            return list(embedding_collection.aggregate(pipeline))

        try:
            raw_results = await run_in_thread(run_aggregate)
        except Exception:
            return []
        doc_ids = [r.get(self._config.embedding_doc_id_field) for r in raw_results if r.get(self._config.embedding_doc_id_field)]

        statutes_collection = self._mongo_client[database][self._config.statutes_collection]

        def run_fetch() -> List[Dict[str, Any]]:
            # Try both statute_id_field (_id) and document_id for lookup; embeddings may reference document_id
            return list(
                statutes_collection.find(
                    {
                        "$or": [
                            {self._config.statute_id_field: {"$in": doc_ids}},
                            {"document_id": {"$in": doc_ids}},
                        ]
                    }
                )
            )

        statute_docs = await run_in_thread(run_fetch) if doc_ids else []
        statute_map: Dict[str, Dict[str, Any]] = {}
        for doc in statute_docs:
            key_id = str(doc.get(self._config.statute_id_field, ""))
            doc_id_key = str(doc.get("document_id", ""))
            if key_id:
                statute_map[key_id] = doc
            if doc_id_key:
                statute_map[doc_id_key] = doc

        candidates = []
        section_field = self._config.statute_section_field  # e.g. "section" (§ 1798.100)
        section_id_field = self._config.statute_section_id_field
        header_field = self._config.statute_chunk_header_field
        for result in raw_results:
            doc_id = str(result.get(self._config.embedding_doc_id_field, ""))
            statute_doc = statute_map.get(doc_id, {})
            # Prefer section (formal citation) from embedding doc; else section_id; else from statute doc
            emb_section = result.get(section_field) or result.get(section_id_field)
            stat_section = statute_doc.get(section_field) or statute_doc.get(section_id_field)
            section_id = str(emb_section or stat_section or "")
            emb_header = result.get(header_field)
            chunk_header = str(emb_header or statute_doc.get(header_field, "") or "")
            # Prefer jurisdiction from embedding doc (e.g. "California") when present; else statute doc
            emb_jur = result.get(self._config.statute_jurisdiction_field) or result.get("jurisdiction")
            cand_jurisdiction = str(statute_doc.get(self._config.statute_jurisdiction_field) or emb_jur or "")
            candidates.append(
                StatuteCandidate(
                    statute_id=doc_id,
                    jurisdiction=cand_jurisdiction,
                    title=str(statute_doc.get(self._config.statute_title_field, "")),
                    section_id=section_id,
                    chunk_text=safe_truncate(
                        str(result.get(self._config.embedding_text_field, "")),
                        self._config.max_statute_chunk_chars,
                    ),
                    score=float(result.get("score", 0.0)),
                    chunk_id=str(result.get(self._config.embedding_chunk_id_field, "0")),
                    chunk_header_text=chunk_header,
                )
            )
        # Hard-coded jurisdiction: keep only California (not CA).
        want_jurisdiction = normalize_jurisdiction(VECTOR_SEARCH_JURISDICTION)
        if want_jurisdiction:
            candidates = [c for c in candidates if normalize_jurisdiction(c.jurisdiction) == want_jurisdiction]
        return candidates

    async def retrieve_policy_subchunks_for_statute_subchunks(
        self,
        database: str,
        policy_document_id: str,
        applicable_jurisdictions: List[str],
        statute_document_id: Optional[str] = None,
        top_k_per_statute: int = 1,
    ) -> RetrievePolicySubchunksResult:
        """Fetch statute subchunks, vector-search policy subchunks for each, return pairs.

        For each statute subchunk in statute_sub_embeddings (filtered by jurisdiction,
        optionally statute_document_id), runs $vectorSearch on policy_sub_embeddings
        using the statute embedding, filtered by policy_document_id.
        """
        statute_coll = self._mongo_client[database][self._config.statute_sub_embeddings_collection]
        policy_coll = self._mongo_client[database][self._config.policy_sub_embeddings_collection]
        vec_path = self._config.embedding_vector_field
        idx_name = self._config.vector_index_name
        jur_field = self._config.statute_jurisdiction_field
        doc_id_field = self._config.policy_document_id_field

        # Build jurisdiction filter
        all_jur_values: List[str] = []
        for j in applicable_jurisdictions or []:
            vals = jurisdiction_filter_values(normalize_jurisdiction(j))
            all_jur_values.extend(vals)
        if not all_jur_values:
            all_jur_values = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)

        statute_filter: Dict[str, Any] = {
            jur_field: {"$in": all_jur_values} if len(all_jur_values) > 1 else all_jur_values[0]
        }
        if statute_document_id:
            statute_filter["document_id"] = statute_document_id

        def fetch_statute_subchunks() -> List[Dict[str, Any]]:
            return list(statute_coll.find(statute_filter))

        statute_docs = await run_in_thread(fetch_statute_subchunks)
        if not statute_docs:
            return RetrievePolicySubchunksResult(pairs=[], statute_subchunks_considered=0)

        policy_filter: Dict[str, Any] = {doc_id_field: policy_document_id}
        pairs: List[SubchunkPair] = []

        for stat_doc in statute_docs:
            query_vector = stat_doc.get(vec_path)
            if not query_vector or not isinstance(query_vector, list):
                continue
            try:
                qv = [float(x) for x in query_vector]
            except (TypeError, ValueError):
                continue

            # Prefer filter in $vectorSearch when index supports it (searches only this policy's chunks).
            # Fallback: no filter, $match after, with higher limit to avoid missing policy chunks.
            # MongoDB requires limit <= numCandidates.
            _limit = max(500, top_k_per_statute * 500)
            _num_cand = max(1000, top_k_per_statute * 100, _limit)
            _vs_base = {"index": idx_name, "path": vec_path, "queryVector": qv, "numCandidates": _num_cand, "limit": _limit}
            pipeline_with_filter = [
                {"$vectorSearch": {**_vs_base, "filter": policy_filter}},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$limit": top_k_per_statute},
            ]
            pipeline_fallback = [
                {"$vectorSearch": _vs_base},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$match": policy_filter},
                {"$limit": top_k_per_statute},
            ]

            def run_search(pipe) -> List[Dict[str, Any]]:
                return list(policy_coll.aggregate(pipe))

            try:
                policy_results = await run_in_thread(lambda p=pipeline_with_filter: run_search(p))
            except Exception:
                policy_results = await run_in_thread(lambda p=pipeline_fallback: run_search(p))

            if not policy_results:
                continue
            best = policy_results[0]
            score = float(best.get("score", 0.0))
            policy_doc = {k: v for k, v in best.items() if k != vec_path}
            if "_id" in policy_doc:
                policy_doc["_id"] = str(policy_doc["_id"])
            pairs.append(
                SubchunkPair(statute_doc=stat_doc, policy_doc=policy_doc, score=score)
            )

        return RetrievePolicySubchunksResult(
            pairs=pairs,
            statute_subchunks_considered=len(statute_docs),
        )

    async def retrieve_policy_chunks_for_statute_chunks(
        self,
        database: str,
        policy_document_id: str,
        applicable_jurisdictions: List[str],
        statute_document_id: Optional[str] = None,
        top_k_per_statute: int = 1,
    ) -> RetrievePolicyChunksResult:
        """Fetch statute chunks from statute_embeddings, vector-search policy_embeddings for each, return pairs.

        Chunk-level (v2): uses statute_embeddings and policy_embeddings instead of subchunk collections.
        """
        statute_coll = self._mongo_client[database][self._config.statute_embeddings_collection]
        policy_coll = self._mongo_client[database][self._config.policy_embeddings_collection]
        vec_path = self._config.embedding_vector_field
        idx_name = self._config.vector_index_name
        jur_field = self._config.statute_jurisdiction_field
        doc_id_field = self._config.policy_document_id_field

        all_jur_values: List[str] = []
        for j in applicable_jurisdictions or []:
            vals = jurisdiction_filter_values(normalize_jurisdiction(j))
            all_jur_values.extend(vals)
        if not all_jur_values:
            all_jur_values = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)

        statute_filter: Dict[str, Any] = {
            jur_field: {"$in": all_jur_values} if len(all_jur_values) > 1 else all_jur_values[0]
        }
        if statute_document_id:
            statute_filter["document_id"] = statute_document_id

        def fetch_statute_chunks() -> List[Dict[str, Any]]:
            return list(statute_coll.find(statute_filter))

        statute_docs = await run_in_thread(fetch_statute_chunks)
        if not statute_docs:
            return RetrievePolicyChunksResult(pairs=[], statute_chunks_considered=0)

        policy_filter: Dict[str, Any] = {doc_id_field: policy_document_id}
        pairs: List[ChunkPair] = []

        for stat_doc in statute_docs:
            query_vector = stat_doc.get(vec_path)
            if not query_vector or not isinstance(query_vector, list):
                continue
            try:
                qv = [float(x) for x in query_vector]
            except (TypeError, ValueError):
                continue

            _limit = max(500, top_k_per_statute * 500)
            _num_cand = max(1000, top_k_per_statute * 100, _limit)  # MongoDB requires limit <= numCandidates
            _vs_base = {"index": idx_name, "path": vec_path, "queryVector": qv, "numCandidates": _num_cand, "limit": _limit}
            pipeline_with_filter = [
                {"$vectorSearch": {**_vs_base, "filter": policy_filter}},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$limit": top_k_per_statute},
            ]
            pipeline_fallback = [
                {"$vectorSearch": _vs_base},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$match": policy_filter},
                {"$limit": top_k_per_statute},
            ]

            def run_search(pipe) -> List[Dict[str, Any]]:
                return list(policy_coll.aggregate(pipe))

            try:
                policy_results = await run_in_thread(lambda p=pipeline_with_filter: run_search(p))
            except Exception:
                policy_results = await run_in_thread(lambda p=pipeline_fallback: run_search(p))

            if not policy_results:
                continue
            best = policy_results[0]
            score = float(best.get("score", 0.0))
            policy_doc = {k: v for k, v in best.items() if k != vec_path}
            if "_id" in policy_doc:
                policy_doc["_id"] = str(policy_doc["_id"])
            pairs.append(
                ChunkPair(statute_doc=stat_doc, policy_doc=policy_doc, score=score)
            )

        return RetrievePolicyChunksResult(
            pairs=pairs,
            statute_chunks_considered=len(statute_docs),
        )

    async def retrieve_statute_policy_pairs_v3(
        self,
        database: str,
        policy_document_id: str,
        applicable_jurisdictions: List[str],
        statute_document_id: Optional[str] = None,
        top_k: int = 5,
        num_candidates: int = 50,
        score_threshold: float = 0.70,
        num_rows: Optional[int] = None,
    ) -> List[StatutePolicyPairV3]:
        """V3 v1 design: Fetch statute subchunks, vector-search policy subchunks, return pairs with parent context.

        Uses statute_sub_embeddings and policy_sub_embeddings. Fetches parent chunks for context.
        Filters matches below score_threshold.
        """
        statute_coll = self._mongo_client[database][self._config.statute_sub_embeddings_collection]
        policy_coll = self._mongo_client[database][self._config.policy_sub_embeddings_collection]
        statute_parent_coll = self._mongo_client[database][self._config.statute_embeddings_collection]
        policy_parent_coll = self._mongo_client[database][self._config.policy_embeddings_collection]
        vec_path = self._config.embedding_vector_field
        # Use vector_index_name (same as retrieve_policy_subchunks_for_statute_subchunks)
        # so policy_sub_embeddings $vectorSearch works
        idx_name = self._config.vector_index_name
        jur_field = self._config.statute_jurisdiction_field
        doc_id_field = self._config.policy_document_id_field
        stat_sub_text = self._config.statute_subchunk_text_field or "subchunk_text"
        pol_sub_text = self._config.policy_subchunk_text_field or "subchunk_text"
        pol_chunk_text = self._config.policy_chunk_text_field or "chunk_text"
        stat_chunk_text = self._config.statute_chunk_text_field or "chunk_text"
        emb_text = self._config.embedding_text_field

        all_jur_values: List[str] = []
        for j in applicable_jurisdictions or []:
            vals = jurisdiction_filter_values(normalize_jurisdiction(j))
            all_jur_values.extend(vals)
        if not all_jur_values:
            all_jur_values = jurisdiction_filter_values(VECTOR_SEARCH_JURISDICTION)

        statute_filter: Dict[str, Any] = {
            jur_field: {"$in": all_jur_values} if len(all_jur_values) > 1 else all_jur_values[0]
        }
        if statute_document_id:
            statute_filter["document_id"] = statute_document_id

        def fetch_statute_subchunks() -> List[Dict[str, Any]]:
            cursor = statute_coll.find(
                statute_filter,
                {
                    vec_path: 1,
                    emb_text: 1,
                    stat_sub_text: 1,
                    "text": 1,
                    "statute_reference": 1,
                    jur_field: 1,
                    "_id": 1,
                    "parent_chunk_id": 1,
                },
            )
            items = list(cursor)
            if num_rows is not None:
                items = items[:num_rows]
            return items

        statute_docs = await run_in_thread(fetch_statute_subchunks)
        if not statute_docs:
            return []

        policy_filter: Dict[str, Any] = {doc_id_field: policy_document_id}
        pairs: List[StatutePolicyPairV3] = []

        for stat_doc in statute_docs:
            query_vector = stat_doc.get(vec_path) or stat_doc.get("embedding")
            if not query_vector or not isinstance(query_vector, list):
                continue
            try:
                qv = [float(x) for x in query_vector]
            except (TypeError, ValueError):
                continue

            _limit = max(500, top_k * 500)
            _num_cand = max(num_candidates, _limit)
            _vs_base = {"index": idx_name, "path": vec_path, "queryVector": qv, "numCandidates": _num_cand, "limit": _limit}
            _proj = {"$project": {"text": 1, "score": 1, "section_id": 1, "chunk_index": 1, pol_sub_text: 1, "parent_chunk_id": 1}}
            pipeline_with_filter = [
                {"$vectorSearch": {**_vs_base, "filter": policy_filter}},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$limit": top_k},
                _proj,
            ]
            pipeline_fallback = [
                {"$vectorSearch": _vs_base},
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$match": policy_filter},
                {"$limit": top_k},
                _proj,
            ]

            def run_v3_search(pipe) -> List[Dict[str, Any]]:
                return list(policy_coll.aggregate(pipe))

            try:
                policy_results = await run_in_thread(lambda p=pipeline_with_filter: run_v3_search(p))
            except Exception:
                policy_results = await run_in_thread(lambda p=pipeline_fallback: run_v3_search(p))

            matches: List[PolicyMatch] = []
            above_threshold = 0
            policy_parent_ids: List[Any] = []
            for r in policy_results:
                score = float(r.get("score", 0.0))
                txt = (r.get(pol_sub_text) or r.get("text") or r.get(pol_chunk_text) or "").strip()
                section_id = str(r.get("section_id") or r.get("chunk_index") or "")
                parent_id = r.get("parent_chunk_id")
                if parent_id is not None:
                    policy_parent_ids.append(parent_id)
                if score >= score_threshold:
                    above_threshold += 1
                matches.append(
                    PolicyMatch(
                        text=txt,
                        score=score,
                        section_id=section_id or None,
                        parent_context=None,
                    )
                )

            # Fetch policy parent chunks
            policy_parent_map: Dict[str, str] = {}
            if policy_parent_ids:
                def fetch_policy_parents() -> Dict[str, str]:
                    parent_docs = list(
                        policy_parent_coll.find(
                            {"_id": {"$in": policy_parent_ids}},
                            {"_id": 1, pol_chunk_text: 1, "text": 1},
                        )
                    )
                    out: Dict[str, str] = {}
                    for d in parent_docs:
                        pid = d.get("_id")
                        if pid is not None:
                            txt = (d.get(pol_chunk_text) or d.get("text") or "").strip()
                            out[str(pid)] = txt
                    return out
                policy_parent_map = await run_in_thread(fetch_policy_parents)
                for i, m in enumerate(matches):
                    pid = policy_results[i].get("parent_chunk_id") if i < len(policy_results) else None
                    if pid is not None:
                        m.parent_context = policy_parent_map.get(str(pid), "")

            # Fetch statute parent chunk
            statute_parent_context = ""
            stat_parent_id = stat_doc.get("parent_chunk_id")
            if stat_parent_id is not None:
                def fetch_statute_parent() -> str:
                    doc = statute_parent_coll.find_one(
                        {"_id": stat_parent_id},
                        {stat_chunk_text: 1, "text": 1, emb_text: 1},
                    )
                    if not doc:
                        return ""
                    return (doc.get(stat_chunk_text) or doc.get(emb_text) or doc.get("text") or "").strip()
                statute_parent_context = await run_in_thread(fetch_statute_parent)

            pairs.append(
                StatutePolicyPairV3(
                    statute_doc=stat_doc,
                    policy_matches=matches,
                    above_threshold_count=above_threshold,
                    statute_parent_context=statute_parent_context or None,
                )
            )

        return pairs
