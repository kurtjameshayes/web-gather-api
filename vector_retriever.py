"""MongoDB vector search wrapper for statute retrieval."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Debug log path: project .cursor/debug.log (not ~/.cursor)
_DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor")
_DEBUG_LOG_PATH = os.path.join(_DEBUG_LOG_DIR, "debug.log")

# Jurisdiction for vector search (override all callers). Configurable via env for DB values like "CA".
VECTOR_SEARCH_JURISDICTION = (os.getenv("COMPLIANCE_VECTOR_SEARCH_JURISDICTION") or "California").strip() or "California"


async def _run_in_thread(func):
    """Run sync function in a thread (Python 3.8 compat: asyncio.to_thread added in 3.9)."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


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

        # #region agent log
        try:
            import json as _json
            os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
            eff_jur = (jurisdiction or "").strip() or VECTOR_SEARCH_JURISDICTION
            fv = jurisdiction_filter_values(normalize_jurisdiction(eff_jur)) if eff_jur else [VECTOR_SEARCH_JURISDICTION]
            with open(_DEBUG_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve", "message": "retrieval path", "data": {"use_embeddings_collection": self._config.use_embeddings_collection, "jurisdiction": jurisdiction, "effective_filter_values": fv[:5]}, "hypothesisId": "H1", "runId": "post-fix"}) + "\n")
        except Exception:
            pass
        # #endregion
        if self._config.use_embeddings_collection:
            results = await self._retrieve_from_embeddings(
                database, query_vector, jurisdiction, statute_corpus_id, top_k
            )
        else:
            results = await self._retrieve_from_statutes(
                database, query_vector, jurisdiction, statute_corpus_id, top_k
            )
        # #region agent log
        try:
            import json as _json
            os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
            with open(_DEBUG_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve", "message": "retrieval result count", "data": {"count": len(results)}, "hypothesisId": "H1"}) + "\n")
        except Exception:
            pass
        # #endregion
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

        raw_results = await _run_in_thread(run_aggregate)
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

        # #region agent log
        _agg_err = None
        raw_results: List[Dict[str, Any]] = []
        try:
            raw_results = await _run_in_thread(run_aggregate)
        except Exception as _e:
            _agg_err = str(_e)
        try:
            import json as _json
            os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
            with open(_DEBUG_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:_retrieve_from_embeddings", "message": "after vectorSearch", "data": {"database": database, "embeddings_collection": self._config.embeddings_collection, "vector_index_name": self._config.vector_index_name, "path": self._config.embedding_vector_field, "query_vector_dim": len(query_vector) if query_vector else 0, "filter": filter_doc, "jurisdiction_requested": jurisdiction, "raw_count": len(raw_results), "aggregate_error": _agg_err}, "hypothesisId": "H2"}) + "\n")
        except Exception:
            pass
        # #endregion
        if _agg_err:
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

        statute_docs = await _run_in_thread(run_fetch) if doc_ids else []
        # #region agent log
        try:
            import json as _json
            _sample_id = str(doc_ids[0]) if doc_ids else None
            os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
            with open(_DEBUG_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:_retrieve_from_embeddings", "message": "after statute lookup", "data": {"doc_ids_count": len(doc_ids), "statute_docs_count": len(statute_docs), "sample_doc_id": _sample_id}, "hypothesisId": "H2"}) + "\n")
        except Exception:
            pass
        # #endregion
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
    ) -> List[SubchunkPair]:
        """Fetch statute subchunks, vector-search policy subchunks for each, return pairs.

        For each statute subchunk in statute_sub_embeddings (filtered by jurisdiction,
        optionally statute_document_id), runs $vectorSearch on policy_sub_embeddings
        using the statute embedding, filtered by policy_document_id.
        """
        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve_policy_subchunks", "message": "entry", "data": {"database": database, "stat_coll": self._config.statute_sub_embeddings_collection, "pol_coll": self._config.policy_sub_embeddings_collection}, "hypothesisId": "H1"}) + "\n")
        except Exception:
            pass
        # #endregion

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

        statute_docs = await _run_in_thread(fetch_statute_subchunks)
        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve_policy_subchunks", "message": "after_fetch_statute", "data": {"statute_count": len(statute_docs), "statute_filter": statute_filter}, "hypothesisId": "H1,H5"}) + "\n")
        except Exception:
            pass
        # #endregion
        if not statute_docs:
            return []

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

            # $vectorSearch filter requires indexed fields; document_id may not be in index.
            # Run vector search without filter, then $match by document_id in pipeline.
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": idx_name,
                        "path": vec_path,
                        "queryVector": qv,
                        "numCandidates": max(500, top_k_per_statute * 50),
                        "limit": 100,
                    }
                },
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$match": policy_filter},
                {"$limit": top_k_per_statute},
            ]

            def run_search() -> List[Dict[str, Any]]:
                return list(policy_coll.aggregate(pipeline))

            try:
                policy_results = await _run_in_thread(run_search)
            except Exception as agg_err:
                # #region agent log
                try:
                    import json as _json
                    with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                        _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve_policy_subchunks", "message": "aggregate_error", "data": {"error": str(agg_err), "index": idx_name, "path": vec_path}, "hypothesisId": "H3,H5"}) + "\n")
                except Exception:
                    pass
                # #endregion
                raise

            # #region agent log
            if not policy_results and statute_docs:
                try:
                    import json as _json
                    with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                        _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve_policy_subchunks", "message": "no_policy_results", "data": {"policy_filter": policy_filter}, "hypothesisId": "H3", "runId": "post-fix"}) + "\n")
                except Exception:
                    pass
            # #endregion
            if not policy_results:
                continue
            best = policy_results[0]
            score = float(best.get("score", 0.0))
            # Exclude raw vector from policy_doc for response
            policy_doc = {k: v for k, v in best.items() if k != vec_path}
            if "_id" in policy_doc:
                policy_doc["_id"] = str(policy_doc["_id"])
            pairs.append(
                SubchunkPair(statute_doc=stat_doc, policy_doc=policy_doc, score=score)
            )

        # #region agent log
        try:
            import json as _json
            with open("/Users/kurthayes/Dev/AI/web-gather-api/.cursor/debug.log", "a") as _f:
                _f.write(_json.dumps({"timestamp": __import__("time").time() * 1000, "location": "vector_retriever.py:retrieve_policy_subchunks", "message": "return", "data": {"pairs_count": len(pairs)}, "hypothesisId": "H3", "runId": "post-fix"}) + "\n")
        except Exception:
            pass
        # #endregion
        return pairs
