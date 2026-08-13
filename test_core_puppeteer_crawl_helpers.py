"""Regression tests for Puppeteer BFS crawl helpers.

Selenium/Playwright BFS is covered on an open coverage PR. Route tests mock
`puppeteer_crawl` entirely, so same-host filtering, visited dedup, breadth
caps, HTTP error skips, and browser.close cleanup had no direct coverage.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

import core


class _FakePuppeteerPage:
    """Minimal async page matching the pyppeteer API used by the crawler."""

    def __init__(self, pages_by_url: Dict[str, Dict[str, Any]]) -> None:
        self._pages = pages_by_url
        self._url: Optional[str] = None
        self.goto_calls: List[str] = []
        self.setUserAgent = AsyncMock()

    async def goto(self, url: str, options: Optional[dict] = None) -> Any:
        self._url = url
        self.goto_calls.append(url)
        page = self._pages.get(url, {"status": 200, "content": "", "links": [], "title": ""})
        response = MagicMock()
        response.status = page.get("status", 200)
        return response

    async def title(self) -> str:
        return self._pages.get(self._url or "", {}).get("title", "")

    async def evaluate(self, script: str) -> Any:
        page = self._pages.get(self._url or "", {"content": "", "links": []})
        if "a[href]" in script:
            return list(page.get("links", []))
        return page.get("content", "")


def _patch_puppeteer(fake_page: _FakePuppeteerPage) -> MagicMock:
    browser = MagicMock()
    browser.newPage = AsyncMock(return_value=fake_page)
    browser.close = AsyncMock()
    core.pyppeteer.launch = AsyncMock(return_value=browser)
    return browser


def test_puppeteer_crawl_same_host_filter_visited_dedup_and_close() -> None:
    """Puppeteer BFS keeps same-host links, skips visited paths, and closes the browser."""
    start = "https://example.com/start"
    pages_by_url = {
        start: {
            "status": 200,
            "title": "Start",
            "content": "start page",
            "links": [
                "https://example.com/a",
                "https://other.example/x",  # off-host: ignored
                "https://example.com/a?utm=1",  # same path as /a after normalize
                "https://example.com/b",
            ],
        },
        "https://example.com/a": {
            "status": 200,
            "title": "A",
            "content": "page a",
            "links": ["https://example.com/start"],  # already visited
        },
        "https://example.com/b": {
            "status": 200,
            "title": "B",
            "content": "page b",
            "links": [],
        },
    }
    fake_page = _FakePuppeteerPage(pages_by_url)
    browser = _patch_puppeteer(fake_page)

    results = asyncio.run(core._puppeteer_crawl_async(start, depth=2, breadth=10))

    urls = [p["url"] for p in results]
    assert urls == [start, "https://example.com/a", "https://example.com/b"]
    assert all(p["markdown"] for p in results)
    assert "https://other.example/x" not in fake_page.goto_calls
    assert fake_page.goto_calls.count("https://example.com/a") == 1
    assert "https://example.com/a?utm=1" not in fake_page.goto_calls
    browser.close.assert_awaited_once()


def test_puppeteer_crawl_stops_at_breadth_and_skips_http_errors() -> None:
    """Breadth caps extracted pages; HTTP >=400 pages are skipped without enqueueing."""
    start = "https://example.com/"
    pages_by_url = {
        start: {
            "status": 200,
            "title": "Root",
            "content": "root",
            "links": [
                "https://example.com/ok",
                "https://example.com/bad",
                "https://example.com/extra",
            ],
        },
        "https://example.com/ok": {"status": 200, "title": "OK", "content": "ok", "links": []},
        "https://example.com/bad": {"status": 404, "title": "Missing", "content": "missing", "links": []},
        "https://example.com/extra": {"status": 200, "title": "Extra", "content": "extra", "links": []},
    }
    fake_page = _FakePuppeteerPage(pages_by_url)
    browser = _patch_puppeteer(fake_page)

    results = asyncio.run(core._puppeteer_crawl_async(start, depth=2, breadth=2))

    assert [p["url"] for p in results] == [start, "https://example.com/ok"]
    assert all(p["url"] != "https://example.com/extra" for p in results)
    assert all(p["url"] != "https://example.com/bad" for p in results)
    browser.close.assert_awaited_once()


def test_puppeteer_crawl_closes_browser_after_page_errors() -> None:
    """browser.close still runs when individual page loads raise."""
    start = "https://example.com/boom"

    class _BoomPage:
        setUserAgent = AsyncMock()

        async def goto(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("navigation failed")

        async def title(self) -> str:
            return ""

        async def evaluate(self, script: str) -> Any:
            return ""

    browser = MagicMock()
    browser.newPage = AsyncMock(return_value=_BoomPage())
    browser.close = AsyncMock()
    core.pyppeteer.launch = AsyncMock(return_value=browser)

    results = asyncio.run(core._puppeteer_crawl_async(start, depth=1, breadth=3))

    assert results == []
    browser.close.assert_awaited_once()


def test_puppeteer_crawl_empty_body_is_not_extracted() -> None:
    """A 200 page with no body text is visited but not counted as extracted content."""
    start = "https://example.com/empty"
    fake_page = _FakePuppeteerPage(
        {start: {"status": 200, "title": "", "content": "", "links": []}}
    )
    browser = _patch_puppeteer(fake_page)

    results = asyncio.run(core._puppeteer_crawl_async(start, depth=1, breadth=3))

    assert results == []
    assert fake_page.goto_calls == [start]
    browser.close.assert_awaited_once()
