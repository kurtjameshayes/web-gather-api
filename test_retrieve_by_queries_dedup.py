"""VectorRetriever.retrieve_by_queries keeps the highest score per statute/chunk."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from compliance_config import load_config
from vector_retriever import StatuteCandidate, VectorRetriever


def _candidate(statute_id: str, chunk_id: str, score: float) -> StatuteCandidate:
    return StatuteCandidate(
        statute_id=statute_id,
        jurisdiction="CA",
        title="Title",
        section_id=chunk_id,
        chunk_text="text",
        score=score,
        chunk_id=chunk_id,
    )


def test_retrieve_by_queries_keeps_higher_score_and_sorts_desc() -> None:
    """Duplicate statute_id:chunk_id keys keep the higher score; results sort by score descending."""
    retriever = VectorRetriever(
        mongo_client=MagicMock(),
        embedder=MagicMock(),
        config=load_config(),
        cache=MagicMock(),
    )
    low = _candidate("s1", "c1", 0.2)
    high = _candidate("s1", "c1", 0.9)
    other = _candidate("s2", "c1", 0.5)

    async def fake_retrieve(**kwargs):
        query = kwargs["section_text"]
        if query == "right to delete":
            return [low, other]
        return [high]

    retriever.retrieve = AsyncMock(side_effect=fake_retrieve)
    result = asyncio.run(
        retriever.retrieve_by_queries(
            database="privacy-compliance",
            jurisdiction="CA",
            queries=["right to delete", "right to know"],
            top_k_per_query=3,
            statute_corpus_id="corpus-1",
        )
    )

    assert [c.statute_id for c in result] == ["s1", "s2"]
    assert result[0].score == 0.9
    assert result[1].score == 0.5
    assert retriever.retrieve.await_count == 2
    first_kwargs = retriever.retrieve.await_args_list[0].kwargs
    assert first_kwargs["statute_corpus_id"] == "corpus-1"
    assert first_kwargs["top_k"] == 3
    assert first_kwargs["jurisdiction"] == "CA"


def test_retrieve_by_queries_equal_score_keeps_first() -> None:
    """A later equal score does not replace the first seen candidate."""
    retriever = VectorRetriever(
        mongo_client=MagicMock(),
        embedder=MagicMock(),
        config=load_config(),
        cache=MagicMock(),
    )
    first = _candidate("s1", "c1", 0.4)
    first.chunk_text = "first"
    second = _candidate("s1", "c1", 0.4)
    second.chunk_text = "second"

    async def fake_retrieve(**kwargs):
        return [first] if kwargs["section_text"] == "q1" else [second]

    retriever.retrieve = AsyncMock(side_effect=fake_retrieve)
    result = asyncio.run(
        retriever.retrieve_by_queries(
            database="db",
            jurisdiction="CA",
            queries=["q1", "q2"],
            top_k_per_query=1,
        )
    )

    assert len(result) == 1
    assert result[0].chunk_text == "first"
