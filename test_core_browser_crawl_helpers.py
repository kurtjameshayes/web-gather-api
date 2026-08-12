"""Regression tests for Selenium/Playwright BFS crawl helpers.

Route tests mock these helpers entirely. These unit tests exercise same-host
filtering, visited dedup, breadth limits, and driver/browser cleanup with
mocked browser APIs so regressions in crawl graph behavior are caught.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["pyppeteer"] = MagicMock()

import core


def _selenium_execute_script_factory(
    *,
    pages_by_url: Dict[str, Dict[str, Any]],
    current_url_holder: List[str],
) -> Any:
    """Build an execute_script side_effect that returns status/content/links."""

    def _execute_script(script: str) -> Any:
        url = current_url_holder[0]
        page = pages_by_url.get(url, {"status": 200, "content": "", "links": []})
        if "performance" in script and "responseStatus" in script:
            return page.get("status", 200)
        if "querySelectorAll('a[href]')" in script or "a[href]" in script:
            return list(page.get("links", []))
        # Text extraction script
        return page.get("content", "")

    return _execute_script


def test_selenium_crawl_same_host_filter_visited_dedup_and_cleanup() -> None:
    """Selenium BFS keeps same-host links, skips visited URLs, and always quits."""
    start = "https://example.com/start"
    pages_by_url = {
        start: {
            "status": 200,
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
            "content": "page a",
            "links": ["https://example.com/start"],  # already visited
        },
        "https://example.com/b": {
            "status": 200,
            "content": "page b",
            "links": [],
        },
    }
    current_url_holder: List[str] = [""]

    driver = MagicMock()
    driver.title = "Example"
    driver.quit = MagicMock()

    def _get(url: str) -> None:
        current_url_holder[0] = url

    driver.get.side_effect = _get
    driver.execute_script.side_effect = _selenium_execute_script_factory(
        pages_by_url=pages_by_url,
        current_url_holder=current_url_holder,
    )

    mock_webdriver = MagicMock()
    mock_webdriver.Chrome.return_value = driver
    mock_options_mod = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "selenium": MagicMock(webdriver=mock_webdriver),
            "selenium.webdriver": mock_webdriver,
            "selenium.webdriver.chrome": MagicMock(options=mock_options_mod),
            "selenium.webdriver.chrome.options": mock_options_mod,
        },
    ):
        results = core.selenium_crawl(start, depth=2, breadth=10)

    urls = [p["url"] for p in results]
    assert urls == [start, "https://example.com/a", "https://example.com/b"]
    assert all(p["markdown"] for p in results)
    # Off-host never fetched
    fetched = [c.args[0] for c in driver.get.call_args_list]
    assert "https://other.example/x" not in fetched
    # Dedup: /a with query string shares normalized path with /a
    assert fetched.count("https://example.com/a") == 1
    assert fetched.count("https://example.com/a?utm=1") == 0
    driver.quit.assert_called_once()


def test_selenium_crawl_stops_at_breadth_and_skips_http_errors() -> None:
    """Breadth caps extracted pages; HTTP >=400 pages are skipped without enqueueing."""
    start = "https://example.com/"
    pages_by_url = {
        start: {
            "status": 200,
            "content": "root",
            "links": [
                "https://example.com/ok",
                "https://example.com/bad",
                "https://example.com/extra",
            ],
        },
        "https://example.com/ok": {"status": 200, "content": "ok", "links": []},
        "https://example.com/bad": {"status": 404, "content": "missing", "links": []},
        "https://example.com/extra": {"status": 200, "content": "extra", "links": []},
    }
    current_url_holder: List[str] = [""]

    driver = MagicMock()
    driver.title = "T"
    driver.quit = MagicMock()
    driver.get.side_effect = lambda url: current_url_holder.__setitem__(0, url)
    driver.execute_script.side_effect = _selenium_execute_script_factory(
        pages_by_url=pages_by_url,
        current_url_holder=current_url_holder,
    )

    mock_webdriver = MagicMock()
    mock_webdriver.Chrome.return_value = driver
    mock_options_mod = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "selenium": MagicMock(webdriver=mock_webdriver),
            "selenium.webdriver": mock_webdriver,
            "selenium.webdriver.chrome": MagicMock(options=mock_options_mod),
            "selenium.webdriver.chrome.options": mock_options_mod,
        },
    ):
        results = core.selenium_crawl(start, depth=2, breadth=2)

    assert [p["url"] for p in results] == [start, "https://example.com/ok"]
    # bad/extra never become extracted pages once breadth is exhausted
    assert all(p["url"] != "https://example.com/extra" for p in results)
    assert all(p["url"] != "https://example.com/bad" for p in results)
    driver.quit.assert_called_once()


def test_selenium_crawl_quits_driver_on_launch_success_even_if_empty() -> None:
    """Driver.quit runs in finally even when the start page yields no content."""
    driver = MagicMock()
    driver.title = ""
    driver.get = MagicMock()
    driver.execute_script.side_effect = [200, ""]  # status ok, empty body
    driver.quit = MagicMock()

    mock_webdriver = MagicMock()
    mock_webdriver.Chrome.return_value = driver
    mock_options_mod = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "selenium": MagicMock(webdriver=mock_webdriver),
            "selenium.webdriver": mock_webdriver,
            "selenium.webdriver.chrome": MagicMock(options=mock_options_mod),
            "selenium.webdriver.chrome.options": mock_options_mod,
        },
    ):
        results = core.selenium_crawl("https://example.com/empty", depth=1, breadth=3)

    assert results == []
    driver.quit.assert_called_once()


class _FakePlaywrightPage:
    def __init__(self, pages_by_url: Dict[str, Dict[str, Any]]) -> None:
        self._pages = pages_by_url
        self._url: Optional[str] = None
        self.goto_calls: List[str] = []

    async def goto(self, url: str, wait_until: str = "", timeout: int = 0) -> Any:
        self._url = url
        self.goto_calls.append(url)
        page = self._pages.get(url, {"status": 200, "content": "", "links": [], "title": ""})
        status = page.get("status", 200)
        response = MagicMock()
        response.status = status
        return response

    async def title(self) -> str:
        return self._pages.get(self._url or "", {}).get("title", "")

    async def evaluate(self, script: str) -> Any:
        page = self._pages.get(self._url or "", {"content": "", "links": []})
        if "a[href]" in script:
            return list(page.get("links", []))
        return page.get("content", "")


def test_playwright_crawl_async_same_host_breadth_and_browser_close() -> None:
    """Playwright BFS filters off-host links, respects breadth, and closes browser."""
    start = "https://docs.example.com/"
    pages_by_url = {
        start: {
            "status": 200,
            "title": "Docs",
            "content": "index",
            "links": [
                "https://docs.example.com/guide",
                "https://evil.example.com/phish",
                "https://docs.example.com/api",
            ],
        },
        "https://docs.example.com/guide": {
            "status": 200,
            "title": "Guide",
            "content": "guide body",
            "links": ["https://docs.example.com/api"],
        },
        "https://docs.example.com/api": {
            "status": 200,
            "title": "API",
            "content": "api body",
            "links": [],
        },
    }
    fake_page = _FakePlaywrightPage(pages_by_url)

    browser = MagicMock()
    browser.close = AsyncMock()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=fake_page)
    browser.new_context = AsyncMock(return_value=context)

    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    playwright_cm = MagicMock()
    playwright_cm.__aenter__ = AsyncMock(
        return_value=MagicMock(chromium=chromium)
    )
    playwright_cm.__aexit__ = AsyncMock(return_value=None)

    async_playwright = MagicMock(return_value=playwright_cm)

    with patch.dict(
        sys.modules,
        {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=async_playwright)},
    ):
        # Re-import path used inside the function: from playwright.async_api import async_playwright
        with patch("playwright.async_api.async_playwright", async_playwright):
            results = asyncio.run(
                core._playwright_crawl_async(start, depth=2, breadth=2)
            )

    assert [p["url"] for p in results] == [start, "https://docs.example.com/guide"]
    assert "https://evil.example.com/phish" not in fake_page.goto_calls
    # breadth=2 stops before fetching /api even though it was queued
    assert "https://docs.example.com/api" not in [p["url"] for p in results]
    browser.close.assert_awaited_once()


def test_playwright_crawl_async_closes_browser_after_page_errors() -> None:
    """Browser.close still runs when individual page loads raise."""
    start = "https://example.com/boom"

    class _BoomPage:
        async def goto(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("navigation failed")

        async def title(self) -> str:
            return ""

        async def evaluate(self, script: str) -> Any:
            return ""

    browser = MagicMock()
    browser.close = AsyncMock()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=_BoomPage())
    browser.new_context = AsyncMock(return_value=context)
    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    playwright_cm = MagicMock()
    playwright_cm.__aenter__ = AsyncMock(return_value=MagicMock(chromium=chromium))
    playwright_cm.__aexit__ = AsyncMock(return_value=None)
    async_playwright = MagicMock(return_value=playwright_cm)

    with patch.dict(
        sys.modules,
        {"playwright": MagicMock(), "playwright.async_api": MagicMock(async_playwright=async_playwright)},
    ):
        with patch("playwright.async_api.async_playwright", async_playwright):
            results = asyncio.run(
                core._playwright_crawl_async(start, depth=1, breadth=3)
            )

    assert results == []
    browser.close.assert_awaited_once()
