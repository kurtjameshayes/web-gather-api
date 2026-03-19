"""Example of tool calling (function calling) in a search workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class ToolCall:
    """Represents a model-requested tool call."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """Represents a single web search result."""

    title: str
    url: str
    snippet: str


ToolHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def build_search_prompt(query: str) -> str:
    """Build a prompt that instructs a model to call the web search tool."""

    return (
        "You are an assistant that can call tools.\n"
        "If a user asks for up-to-date information, call the tool:\n"
        "web_search(query: string, max_results: int)\n\n"
        f"User request: {query}\n"
        "Decide whether to call the tool and provide the function call if so."
    )


def mock_model_response(prompt: str) -> ToolCall:
    """Return a mock tool call based on the prompt content."""

    if "up-to-date" in prompt or "latest" in prompt:
        return ToolCall(
            name="web_search",
            arguments={"query": "latest Python releases", "max_results": 3},
        )
    return ToolCall(name="respond", arguments={"text": "No tool needed."})


def web_search_tool(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Simulate a web search tool that returns structured results."""

    query = str(arguments.get("query", ""))
    max_results = int(arguments.get("max_results", 3))
    results = [
        {
            "title": "Python.org Downloads",
            "url": "https://www.python.org/downloads/",
            "snippet": f"Official Python downloads page for query: {query}.",
        },
        {
            "title": "Python Release Notes",
            "url": "https://docs.python.org/3/whatsnew/",
            "snippet": "What's new in the latest Python versions.",
        },
    ]
    return {"results": results[:max_results]}


def parse_search_results(payload: Mapping[str, Any]) -> List[SearchResult]:
    """Parse tool output into a list of SearchResult values."""

    parsed: List[SearchResult] = []
    for item in payload.get("results", []):
        parsed.append(
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("snippet", "")),
            )
        )
    return parsed


def execute_tool_call(
    tool_call: ToolCall, registry: Mapping[str, ToolHandler]
) -> Mapping[str, Any]:
    """Execute the requested tool call using a registry."""

    handler = registry.get(tool_call.name)
    if handler is None:
        print("Step 3: Tool not found in registry.")
        return {"error": f"Unknown tool: {tool_call.name}"}
    print("Step 3: Calling the tool with model-provided arguments.")
    return handler(tool_call.arguments)


def web_search_workflow(query: str) -> Sequence[str]:
    """End-to-end example of function calling in a web search workflow."""

    print("Step 1: Build a prompt that allows tool calling.")
    prompt = build_search_prompt(query)
    print("Step 1 result: Prompt ready for the model.")

    print("Step 2: Model decides whether to call a tool.")
    tool_call = mock_model_response(prompt)
    print(
        "Step 2 result: Model requested tool "
        f"{tool_call.name} with args {tool_call.arguments}."
    )
    tool_registry: Dict[str, ToolHandler] = {"web_search": web_search_tool}

    if tool_call.name != "web_search":
        print("Step 3: Model responded directly, no tool call made.")
        return ["Model answered directly without tool use."]

    tool_output = execute_tool_call(tool_call, tool_registry)
    print("Step 4: Tool returned raw results.")
    results = parse_search_results(tool_output)
    print("Step 5: Parsed tool output into structured results.")

    formatted_results = [
        f"{index + 1}. {result.title} - {result.url} ({result.snippet})"
        for index, result in enumerate(results)
    ]
    print("Step 6: Streaming formatted results as they are built.")
    for line in formatted_results:
        print(f"- {line}")
    return formatted_results


def main() -> None:
    """Run the example workflow and print formatted results."""

    query = "Show the latest Python releases"
    web_search_workflow(query)


if __name__ == "__main__":
    main()
