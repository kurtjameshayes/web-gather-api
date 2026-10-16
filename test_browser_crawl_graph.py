"""Link-following rules for the browser crawlers.

Route tests mock playwright_crawl, puppeteer_crawl, and selenium_crawl.
These tests drive the crawlers with fake browsers and pin which URLs are
fetched, which pages are kept, and when a failure stops the branch.
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sentence_transformers", MagicMock())

import core


def _page(
    *,
    status: int | None = 200,
    title: str = "",
    content: str = "",
    links: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "title": title,
        "content": content,
        "links": links or [],
        "error": error,
    }


def _restore_event_loop() -> None:
    asyncio.set_event_loop(asyncio.new_event_loop())


class _Response:
    def __init__(self, status: int | None) -> None:
        self.status = status


class _AsyncPage:
    def __init__(self, site: dict[str, dict[str, Any]], urls: list[str]) -> None:
        self._site = site
        self.urls = urls
        self._url = ""

    async def goto(self, url: str, *args: Any, **kwargs: Any) -> _Response | None:
        self.urls.append(url)
        self._url = url
        spec = self._site.get(url)
        if spec is None:
            return None
        if spec.get("error"):
            raise RuntimeError(spec["error"])
        return _Response(spec.get("status", 200))

    async def title(self) -> str:
        return self._site[self._url].get("title") or ""

    async def setUserAgent(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def evaluate(self, script: str) -> Any:
        spec = self._site[self._url]
        if script == core._EXTRACT_LINKS_JS:
            return list(spec.get("links") or [])
        if script == core._EXTRACT_TEXT_JS:
            return spec.get("content") or ""
        raise AssertionError(f"unexpected evaluate script: {script[:80]}")


class _PlaywrightBrowser:
    def __init__(self, page: _AsyncPage) -> None:
        self._page = page
        self.closed = False

    async def new_context(self, **_kwargs: Any) -> "_PlaywrightBrowser":
        return self

    async def new_page(self) -> _AsyncPage:
        return self._page

    async def close(self) -> None:
        self.closed = True


def _install_playwright(site: dict[str, dict[str, Any]], urls: list[str], holder: dict[str, Any]) -> None:
    page = _AsyncPage(site, urls)
    browser = _PlaywrightBrowser(page)

    class _Chromium:
        async def launch(self, **_kwargs: Any) -> _PlaywrightBrowser:
            holder["browser"] = browser
            return browser

    class _Session:
        def __init__(self) -> None:
            self.chromium = _Chromium()

        async def __aenter__(self) -> "_Session":
            return self

        async def __aexit__(self, *_exc: Any) -> bool:
            return False

    api = types.ModuleType("playwright.async_api")
    api.async_playwright = lambda: _Session()
    package = types.ModuleType("playwright")
    package.__path__ = []  # type: ignore[attr-defined]
    package.async_api = api
    sys.modules["playwright"] = package
    sys.modules["playwright.async_api"] = api
    holder["browser"] = browser


def _run_playwright(
    site: dict[str, dict[str, Any]], depth: int, breadth: int
) -> tuple[list[dict], list[str], bool]:
    urls: list[str] = []
    holder: dict[str, Any] = {}
    saved = {
        name: sys.modules.get(name)
        for name in ("playwright", "playwright.async_api")
    }
    try:
        _install_playwright(site, urls, holder)
        pages = core.playwright_crawl(START, depth, breadth)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        _restore_event_loop()
    browser = holder.get("browser")
    return pages, urls, bool(browser and browser.closed)


def _run_puppeteer(
    site: dict[str, dict[str, Any]], depth: int, breadth: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict], list[str], bool]:
    urls: list[str] = []
    page = _AsyncPage(site, urls)
    holder: dict[str, Any] = {}

    class _Browser:
        async def newPage(self) -> _AsyncPage:
            return page

        async def close(self) -> None:
            holder["closed"] = True

    async def launch(**_kwargs: Any) -> _Browser:
        return _Browser()

    monkeypatch.setattr(core.pyppeteer, "launch", launch)
    try:
        pages = core.puppeteer_crawl(START, depth, breadth)
    finally:
        _restore_event_loop()
    return pages, urls, holder.get("closed") is True


class _SeleniumDriver:
    def __init__(self, site: dict[str, dict[str, Any]], urls: list[str]) -> None:
        self._site = site
        self.urls = urls
        self._url = ""
        self.closed = False

    def get(self, url: str) -> None:
        self.urls.append(url)
        self._url = url
        spec = self._site.get(url)
        if spec and spec.get("error"):
            raise RuntimeError(spec["error"])

    def execute_script(self, script: str) -> Any:
        spec = self._site.get(self._url) or {}
        if "responseStatus" in script:
            return spec.get("status", 200)
        if "a[href]" in script:
            return list(spec.get("links") or [])
        if "querySelectorAll('script" in script:
            return spec.get("content") or ""
        raise AssertionError(f"unexpected selenium script: {script[:80]}")

    @property
    def title(self) -> str:
        spec = self._site.get(self._url) or {}
        return spec.get("title") or ""

    def quit(self) -> None:
        self.closed = True


def _run_selenium(
    site: dict[str, dict[str, Any]], depth: int, breadth: int
) -> tuple[list[dict], list[str], bool]:
    urls: list[str] = []
    names = [
        "selenium",
        "selenium.webdriver",
        "selenium.webdriver.chrome",
        "selenium.webdriver.chrome.options",
    ]
    saved = {name: sys.modules.get(name) for name in names}

    selenium = types.ModuleType("selenium")
    selenium.__path__ = []  # type: ignore[attr-defined]
    webdriver = types.ModuleType("selenium.webdriver")
    webdriver.__path__ = []  # type: ignore[attr-defined]
    chrome = types.ModuleType("selenium.webdriver.chrome")
    chrome.__path__ = []  # type: ignore[attr-defined]
    options_mod = types.ModuleType("selenium.webdriver.chrome.options")

    class Options:
        def add_argument(self, _arg: str) -> None:
            return None

    class Chrome(_SeleniumDriver):
        def __init__(self, options: Any = None) -> None:
            super().__init__(site, urls)
            driver_holder.append(self)

    driver_holder: list[_SeleniumDriver] = []
    options_mod.Options = Options
    webdriver.Chrome = Chrome
    chrome.options = options_mod
    selenium.webdriver = webdriver
    sys.modules["selenium"] = selenium
    sys.modules["selenium.webdriver"] = webdriver
    sys.modules["selenium.webdriver.chrome"] = chrome
    sys.modules["selenium.webdriver.chrome.options"] = options_mod
    try:
        pages = core.selenium_crawl(START, depth, breadth)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    used = driver_holder[0]
    return pages, used.urls, used.closed


def _bind_crawlers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Callable]:
    return {
        "playwright": lambda site, depth, breadth: _run_playwright(site, depth, breadth),
        "puppeteer": lambda site, depth, breadth: _run_puppeteer(site, depth, breadth, monkeypatch),
        "selenium": lambda site, depth, breadth: _run_selenium(site, depth, breadth),
    }


@pytest.fixture(params=["playwright", "puppeteer", "selenium"])
def crawl(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    return _bind_crawlers(monkeypatch)[request.param]


START = "https://news.example/start?ref=home"


def _graph() -> dict[str, dict[str, Any]]:
    story = "https://news.example/story?id=1#top"
    return {
        START: _page(
            title="Home",
            content="Home",
            links=[
                story,
                "https://news.example/story?id=2",
                "https://other.example/out",
                "https://news.example:8443/alt",
                "/relative",
                "https://cdn.news.example/asset",
                "https://news.example/start",
                "http://news.example/plain",
            ],
        ),
        story: _page(
            title="Story",
            content="Story",
            links=["https://news.example/grandchild"],
        ),
        "http://news.example/plain": _page(title="Plain", content="Plain"),
        "https://news.example/grandchild": _page(title="Grand", content="Grand"),
    }


def test_crawl_stays_on_host_and_dedups_path(crawl) -> None:
    pages, urls, closed = crawl(_graph(), 2, 10)

    assert urls == [
        START,
        "https://news.example/story?id=1#top",
        "http://news.example/plain",
    ]
    assert [(page["url"], page["title"], page["markdown"]) for page in pages] == [
        (START, "Home", "Home"),
        ("https://news.example/story?id=1#top", "Story", "Story"),
        ("http://news.example/plain", "Plain", "Plain"),
    ]
    assert closed is True


def test_error_status_drops_page_and_does_not_follow_links(crawl) -> None:
    site = {
        START: _page(
            content="Home",
            links=[
                "https://news.example/denied",
                "https://news.example/edge",
                "https://news.example/zero",
            ],
        ),
        "https://news.example/denied": _page(
            status=400,
            content="nope",
            links=["https://news.example/secret"],
        ),
        "https://news.example/edge": _page(status=399, title="Edge", content="edge"),
        "https://news.example/zero": _page(status=0, title="Zero", content="zero"),
        "https://news.example/secret": _page(content="secret"),
    }
    pages, urls, _closed = crawl(site, 3, 10)

    assert urls == [
        START,
        "https://news.example/denied",
        "https://news.example/edge",
        "https://news.example/zero",
    ]
    assert [page["url"] for page in pages] == [
        START,
        "https://news.example/edge",
        "https://news.example/zero",
    ]


def test_null_status_follows_only_on_selenium(monkeypatch: pytest.MonkeyPatch) -> None:
    site = {
        START: _page(
            content="Home",
            links=["https://news.example/missing-status"],
        ),
        "https://news.example/missing-status": _page(
            status=None,
            title="Missing",
            content="missing",
            links=["https://news.example/from-missing"],
        ),
        "https://news.example/from-missing": _page(title="Followed", content="followed"),
    }
    runners = _bind_crawlers(monkeypatch)
    for name, crawl in runners.items():
        _pages, urls, _closed = crawl(site, 3, 10)
        kept = [page["url"] for page in _pages]
        if name == "selenium":
            assert "https://news.example/missing-status" in kept
            assert "https://news.example/from-missing" in urls
        else:
            assert "https://news.example/missing-status" not in kept
            assert "https://news.example/from-missing" not in urls


def test_empty_page_does_not_count_but_its_links_are_followed(crawl) -> None:
    site = {
        START: _page(
            title="Blank",
            content="",
            links=["https://news.example/next", "https://news.example/space"],
        ),
        "https://news.example/next": _page(
            title="Next",
            content="next",
            links=["https://news.example/grandchild"],
        ),
        "https://news.example/space": _page(title="Space", content=" "),
        "https://news.example/grandchild": _page(content="grand"),
    }

    pages, urls, _closed = crawl(site, 2, 10)
    assert urls == [START, "https://news.example/next", "https://news.example/space"]
    assert [(page["url"], page["markdown"]) for page in pages] == [
        ("https://news.example/next", "next"),
        ("https://news.example/space", " "),
    ]

    pages, urls, _closed = crawl(site, 1, 10)
    assert urls == [START]
    assert pages == []

    content_parent = {
        START: _page(content="Home", links=["https://news.example/next"]),
        "https://news.example/next": _page(content="next"),
    }
    pages, urls, _closed = crawl(content_parent, 2, 1)
    assert urls == [START]
    assert [page["url"] for page in pages] == [START]

    pages, urls, _closed = crawl(site, 2, 1)
    assert urls == [START, "https://news.example/next"]
    assert [page["url"] for page in pages] == ["https://news.example/next"]


def test_page_error_continues_and_closes_browser(crawl) -> None:
    site = {
        START: _page(
            title="Home",
            content="Home",
            links=["https://news.example/boom", "https://news.example/ok"],
        ),
        "https://news.example/boom": _page(error="navigation failed"),
        "https://news.example/ok": _page(title="Ok", content="ok"),
    }
    pages, urls, closed = crawl(site, 2, 10)
    assert urls == [START, "https://news.example/boom", "https://news.example/ok"]
    assert [(page["url"], page["markdown"]) for page in pages] == [
        (START, "Home"),
        ("https://news.example/ok", "ok"),
    ]
    assert closed is True
