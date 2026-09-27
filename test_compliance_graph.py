"""Regression tests for the in-repo StateGraph used when langgraph is absent.

Background gap-analysis, health-score, and sub-vector-index jobs all execute
through this runner. These tests pin execution order and failure behavior.
"""
from __future__ import annotations

import asyncio

import pytest

from compliance_graph import END, START, StateGraph


def test_ainvoke_merges_sync_then_async_nodes_without_mutating_input() -> None:
    """Later nodes see earlier updates, and the caller's dict stays unchanged."""
    graph = StateGraph(dict)

    def sync_node(state: dict) -> dict:
        return {"n": state["n"] + 1, "sync": True}

    async def async_node(state: dict) -> dict:
        return {"n": state["n"] + 1, "async": True}

    graph.add_node("sync", sync_node)
    graph.add_node("async", async_node)
    graph.add_edge(START, "sync")
    graph.add_edge("sync", "async")
    graph.add_edge("async", END)

    initial = {"n": 1}
    result = asyncio.run(graph.compile().ainvoke(initial))

    assert initial == {"n": 1}
    assert result == {"n": 3, "sync": True, "async": True}


def test_ainvoke_follows_only_the_first_outgoing_edge() -> None:
    """Fan-out is not executed. A second edge from the same node is ignored."""
    seen: list[str] = []
    graph = StateGraph(dict)

    def mark(name: str):
        def _node(_state: dict) -> dict:
            seen.append(name)
            return {name: True}

        return _node

    graph.add_node("a", mark("a"))
    graph.add_node("b", mark("b"))
    graph.add_node("c", mark("c"))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.set_finish_point("b")
    graph.set_finish_point("c")

    result = asyncio.run(graph.compile().ainvoke({}))

    assert seen == ["a", "b"]
    assert result == {"a": True, "b": True}


def test_ainvoke_stops_when_the_next_node_is_missing() -> None:
    """A dangling edge stops the run and keeps state from the last real node."""
    seen: list[str] = []
    graph = StateGraph(dict)
    graph.add_node("a", lambda _state: seen.append("a") or {"from_a": True})
    graph.add_node("b", lambda _state: seen.append("b") or {"from_b": True})
    graph.set_entry_point("a")
    graph.add_edge("a", "gone")
    graph.add_edge("gone", "b")

    result = asyncio.run(graph.compile().ainvoke({"seed": 1}))

    assert seen == ["a"]
    assert result == {"seed": 1, "from_a": True}


def test_ainvoke_with_no_entry_returns_a_copy_of_the_initial_state() -> None:
    """No START edge does not run nodes and does not return the same dict."""
    graph = StateGraph(dict)
    graph.add_node("a", lambda _state: {"a": True})
    initial = {"keep": 1}

    result = asyncio.run(graph.compile().ainvoke(initial))

    assert result == {"keep": 1}
    assert result is not initial


def test_ainvoke_ignores_non_dict_node_results_and_continues() -> None:
    """A list or None return does not replace state; the next node still runs."""
    graph = StateGraph(dict)
    graph.add_node("weird", lambda _state: ["not", "a", "dict"])
    graph.add_node("empty", lambda _state: None)

    async def later(state: dict) -> dict:
        return {"later": state["seed"] + 1}

    graph.add_node("later", later)
    graph.set_entry_point("weird")
    graph.add_edge("weird", "empty")
    graph.add_edge("empty", "later")
    graph.set_finish_point("later")

    result = asyncio.run(graph.compile().ainvoke({"seed": 4}))

    assert result == {"seed": 4, "later": 5}


def test_ainvoke_propagates_node_exceptions() -> None:
    """Node failures are not swallowed by the runner."""
    graph = StateGraph(dict)

    async def boom(_state: dict) -> dict:
        raise RuntimeError("node failed")

    graph.add_node("boom", boom)
    graph.set_entry_point("boom")
    graph.set_finish_point("boom")

    with pytest.raises(RuntimeError, match="node failed"):
        asyncio.run(graph.compile().ainvoke({}))
