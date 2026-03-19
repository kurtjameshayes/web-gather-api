"""
Lightweight LangGraph-style StateGraph for compliance workflows.
Uses the same StateGraph/node/edge pattern as LangGraph for compatibility.
When Python 3.10+ is available, replace with: from langgraph.graph import StateGraph, START, END
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional, TypedDict

# Graph constants (match langgraph)
START = "__start__"
END = "__end__"


class StateGraph:
    """
    Minimal StateGraph implementation compatible with LangGraph pattern.
    Supports sync and async nodes. Use .compile() then .ainvoke() for async execution.
    """

    def __init__(self, state_schema: type) -> None:
        self._state_schema = state_schema
        self._nodes: Dict[str, Callable] = {}
        self._edges: list = []  # (from, to) or (from, None) for END

    def add_node(self, name: str, fn: Callable) -> None:
        self._nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._edges.append((from_node, to_node))

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self._nodes, self._edges)

    def set_entry_point(self, name: str) -> None:
        self._edges.append((START, name))

    def set_finish_point(self, name: str) -> None:
        self._edges.append((name, END))


class CompiledGraph:
    def __init__(self, nodes: Dict[str, Callable], edges: list) -> None:
        self._nodes = nodes
        self._edges = edges
        self._build_next_map()

    def _build_next_map(self) -> None:
        self._next: Dict[str, list] = {}
        for fr, to in self._edges:
            if fr == START:
                self._next[START] = self._next.get(START, []) + [to]
            else:
                self._next[fr] = self._next.get(fr, []) + [to]

    async def ainvoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph asynchronously. Supports async nodes."""
        state = dict(initial_state)
        current = self._next.get(START, [])
        if not current:
            return state
        current = current[0]

        while current and current != END:
            node_fn = self._nodes.get(current)
            if not node_fn:
                break
            result = node_fn(state)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                state.update(result)
            next_nodes = self._next.get(current, [])
            if not next_nodes:
                break
            current = next_nodes[0] if next_nodes[0] != END else None

        return state
