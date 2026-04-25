"""Tests for the lightweight fallback graph runtime."""
from __future__ import annotations

import asyncio

from compliance_graph import END, START, StateGraph


def test_state_graph_runs_sync_and_async_nodes_in_order() -> None:
    """Compiled graph merges node outputs without mutating the input state."""
    calls: list[str] = []

    def first_node(state: dict[str, object]) -> dict[str, object]:
        calls.append("first")
        return {"visited": [*state["visited"], "first"], "count": 1}

    async def second_node(state: dict[str, object]) -> dict[str, object]:
        calls.append("second")
        return {"visited": [*state["visited"], "second"], "count": state["count"] + 1}

    graph = StateGraph(dict)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    initial_state = {"visited": []}
    result = asyncio.run(graph.compile().ainvoke(initial_state))

    assert result == {"visited": ["first", "second"], "count": 2}
    assert initial_state == {"visited": []}
    assert calls == ["first", "second"]


def test_state_graph_stops_before_missing_node_and_keeps_state() -> None:
    """A bad edge should stop execution without dropping accumulated state."""
    graph = StateGraph(dict)
    graph.add_node("known", lambda state: {"status": "started"})
    graph.add_edge(START, "known")
    graph.add_edge("known", "missing")

    result = asyncio.run(graph.compile().ainvoke({"job_id": "job-1", "status": "pending"}))

    assert result == {"job_id": "job-1", "status": "started"}
