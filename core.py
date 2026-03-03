"""Core endpoints for the Web Gather API.

Includes endpoints: gather, ingest, parse-llm, gap-check, search, vector-search, create-paragraph-sections, create-statute-subsections, create-statute-subtopics, create-policy-subsections, create-chunks, index, and crawl.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import uuid
from urllib.parse import urlparse, urljoin

import numpy as np
import pyppeteer
import requests
from flask import Blueprint, jsonify, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from bson import ObjectId
from bson.errors import InvalidId
from compliance_utils import extract_json_block
from db import (
    DOCUMENTS_COLLECTION,
    PRIVACY_COMPLIANCE_DB,
    WEB_GATHER_DB,
    get_application_embedding_model,
    get_embedding_model_name,
    get_embedding_model_record,
    utc_now,
)

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None
firecrawl_client = None
anthropic_client = None


def _warn_if_database_or_collection_not_found(
    database: str,
    source_collection: str,
    log_prefix: str,
) -> None:
    """Log warnings if database or source collection are not found."""
    if mongo_client is None:
        return
    try:
        db_names = mongo_client.list_database_names()
        if database not in db_names:
            logger.warning(
                "%s - Database %r not found. Available databases: %s",
                log_prefix,
                database,
                sorted(db_names),
            )
            return
        db = mongo_client[database]
        coll_names = db.list_collection_names()
        if source_collection not in coll_names:
            logger.warning(
                "%s - Source collection %r not found in database %r. Available collections: %s",
                log_prefix,
                source_collection,
                database,
                sorted(coll_names),
            )
    except Exception as e:
        logger.warning("%s - Could not verify database/collections: %s", log_prefix, e)


def _get_json_payload_or_error():
    """Get request JSON. Returns (payload, None) on success, or (None, (response, status)) on error."""
    payload = request.get_json(silent=True)
    if payload is not None:
        return (payload if isinstance(payload, dict) else {}, None)
    raw = request.get_data(as_text=True) or ""
    if not raw.strip():
        return ({}, None)
    try:
        return (json.loads(raw), None)
    except json.JSONDecodeError as exc:
        return (None, (jsonify({"error": f"Invalid JSON in request body: {str(exc)}"}), 400))


def calculate_relevance_score(query, title, description):
    """Calculate relevance score based on query term matching.

    Scores content based on how well query terms match the title and description.
    Title matches are weighted higher than description matches.

    Args:
        query: The search query string
        title: The result title
        description: The result description

    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    if not query:
        return 0.0

    # Normalize and tokenize query
    query_lower = query.lower()
    query_terms = set(re.findall(r'\b\w+\b', query_lower))
    if not query_terms:
        return 0.0

    # Normalize title and description
    title_lower = (title or "").lower()
    desc_lower = (description or "").lower()

    # Count term matches
    title_matches = sum(1 for term in query_terms if term in title_lower)
    desc_matches = sum(1 for term in query_terms if term in desc_lower)

    # Calculate weighted score (title matches worth 2x description matches)
    # Max possible score: all terms in title (2 points each) + all in description (1 point each)
    max_score = len(query_terms) * 3  # 2 for title + 1 for description
    actual_score = (title_matches * 2) + desc_matches

    # Normalize to 0-1 range
    score = min(actual_score / max_score, 1.0) if max_score > 0 else 0.0

    # Boost for exact phrase match
    if query_lower in title_lower:
        score = min(score + 0.2, 1.0)
    elif query_lower in desc_lower:
        score = min(score + 0.1, 1.0)

    return round(score, 3)

# Model cache for sentence transformers
_model_cache = {}

core_bp = Blueprint("core", __name__)


def init_core(mongo, firecrawl, anthropic):
    """Initialize the core module with required clients."""
    global mongo_client, firecrawl_client, anthropic_client
    mongo_client = mongo
    firecrawl_client = firecrawl
    anthropic_client = anthropic


def get_model(model_name: str) -> SentenceTransformer:
    """Get or create a cached sentence transformer model."""
    model = _model_cache.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    return model


def normalize_results(raw_result):
    """Normalize Firecrawl results to a list format."""
    logger.info("normalize_results: input type: %s", type(raw_result))

    # Handle Firecrawl CrawlJob response (has .data attribute)
    if hasattr(raw_result, "data") and isinstance(raw_result.data, list):
        logger.info("normalize_results: returning .data list with %d items", len(raw_result.data))
        return raw_result.data
    # Handle Firecrawl SearchData response (has .web attribute)
    if hasattr(raw_result, "web") and raw_result.web is not None:
        logger.info("normalize_results: returning .web with %d items", len(raw_result.web))
        return raw_result.web
    # Handle dict responses (legacy compatibility)
    if isinstance(raw_result, dict):
        logger.info("normalize_results: dict keys: %s", list(raw_result.keys()))
        if "data" in raw_result:
            logger.info("normalize_results: returning dict['data'] with %d items", len(raw_result["data"]))
            return raw_result["data"]
        if "results" in raw_result:
            logger.info("normalize_results: returning dict['results']")
            return raw_result["results"]
        if "pages" in raw_result:
            logger.info("normalize_results: returning dict['pages']")
            return raw_result["pages"]
    if isinstance(raw_result, list):
        logger.info("normalize_results: returning list with %d items", len(raw_result))
        return raw_result
    logger.warning("normalize_results: no matching pattern, returning empty list")
    return []


def get_page_attr(page, attr, default=""):
    """Get attribute from page object or dict."""
    if hasattr(page, attr):
        return getattr(page, attr, default) or default
    if isinstance(page, dict):
        return page.get(attr, default) or default
    return default


def combine_pages(pages):
    """Combine multiple pages into a single text."""
    logger.info("combine_pages: Processing %d pages", len(pages))
    combined = []
    for idx, page in enumerate(pages):
        logger.info("combine_pages: Page %d type: %s", idx, type(page))

        # Debug: log all available attributes
        if hasattr(page, '__dict__'):
            logger.info("combine_pages: Page %d attrs: %s", idx, list(vars(page).keys()))
        elif isinstance(page, dict):
            logger.info("combine_pages: Page %d dict keys: %s", idx, list(page.keys()))

        # Handle Firecrawl Document objects (url/title in metadata)
        if hasattr(page, "metadata") and page.metadata:
            url = getattr(page.metadata, "url", "") or ""
            title = getattr(page.metadata, "title", "") or ""
            logger.info("combine_pages: Page %d metadata - url: %s, title: %s", idx, url[:50] if url else "None", title[:50] if title else "None")
        else:
            url = get_page_attr(page, "url")
            title = get_page_attr(page, "title")
            logger.info("combine_pages: Page %d direct - url: %s, title: %s", idx, url[:50] if url else "None", title[:50] if title else "None")

        # Get content - try markdown first (Firecrawl), then fallbacks
        # Note: Firecrawl Document fields are: markdown, html, raw_html, summary
        markdown_content = get_page_attr(page, "markdown")
        html_content = get_page_attr(page, "html")
        raw_html_content = get_page_attr(page, "raw_html")
        summary_content = get_page_attr(page, "summary")

        logger.info("combine_pages: Page %d content lengths - markdown: %d, html: %d, raw_html: %d, summary: %d",
                   idx,
                   len(markdown_content) if markdown_content else 0,
                   len(html_content) if html_content else 0,
                   len(raw_html_content) if raw_html_content else 0,
                   len(summary_content) if summary_content else 0)

        content = markdown_content or html_content or raw_html_content or summary_content
        if not content:
            logger.warning("combine_pages: Page %d has no extractable content, skipping", idx)
            continue
        header = "Source"
        if title:
            header = f"{title} | Source"
        combined.append(f"{header}: {url}\n{content}".strip())
        logger.info("combine_pages: Page %d added with content length: %d", idx, len(content))

    result = "\n\n".join(combined).strip()
    logger.info("combine_pages: Final combined text length: %d", len(result))
    return result


def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    # Check file extension
    if path_lower.endswith(".pdf"):
        return True
    # Check common PDF query patterns
    if "pdf" in parsed.query.lower():
        return True
    return False


def detect_content_type(url: str) -> str:
    """Make a HEAD request to detect the content type of a URL."""
    try:
        logger.info("Detecting content type for URL: %s", url)
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get("Content-Type", "").lower()
        logger.info("Content-Type detected: %s", content_type)
        return content_type
    except Exception as exc:
        logger.warning("Failed to detect content type for %s: %s", url, exc)
        return ""


# Shared JS for content extraction (used by Playwright, Puppeteer, Selenium)
_EXTRACT_TEXT_JS = """() => {
    const scripts = document.querySelectorAll('script, style, noscript');
    scripts.forEach(el => el.remove());
    const body = document.body;
    if (!body) return '';
    function getText(element) {
        let text = '';
        for (const node of element.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                text += node.textContent.trim() + ' ';
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                const tagName = node.tagName.toLowerCase();
                if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
                    text += '\\n\\n## ' + getText(node) + '\\n\\n';
                } else if (['p', 'div', 'section', 'article'].includes(tagName)) {
                    text += getText(node) + '\\n\\n';
                } else if (tagName === 'li') {
                    text += '- ' + getText(node) + '\\n';
                } else if (tagName === 'br') {
                    text += '\\n';
                } else if (!['script', 'style', 'noscript'].includes(tagName)) {
                    text += getText(node);
                }
            }
        }
        return text;
    }
    return getText(body).replace(/\\n{3,}/g, '\\n\\n').trim();
}"""
_EXTRACT_LINKS_JS = """() => {
    const anchors = document.querySelectorAll('a[href]');
    return Array.from(anchors)
        .map(a => a.href)
        .filter(href => href && href.startsWith('http'));
}"""


async def _playwright_crawl_async(start_url: str, depth: int, breadth: int) -> list[dict]:
    """Crawl pages using Playwright (async implementation).

    Args:
        start_url: The URL to start crawling from
        depth: How deep to follow links (1 = only start page)
        breadth: Maximum number of pages to crawl

    Returns:
        List of page dicts with 'url', 'title', and 'markdown' keys
    """
    from playwright.async_api import async_playwright

    logger.info("Playwright crawl starting: url=%s, depth=%d, breadth=%d", start_url, depth, breadth)

    visited = set()
    pages = []
    to_visit = [(start_url, 0)]
    base_parsed = urlparse(start_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        try:
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            while to_visit and len(pages) < breadth:
                current_url, current_depth = to_visit.pop(0)

                parsed = urlparse(current_url)
                normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if normalized_url in visited:
                    continue
                visited.add(normalized_url)

                logger.info("Playwright crawling: %s (depth=%d)", current_url, current_depth)

                try:
                    response = await page.goto(current_url, wait_until="networkidle", timeout=30000)

                    if not response or response.status >= 400:
                        logger.warning(
                            "Playwright: Failed to load %s (status=%s)",
                            current_url, response.status if response else "no response"
                        )
                        continue

                    title = await page.title() or ""
                    content = await page.evaluate(_EXTRACT_TEXT_JS)

                    if content:
                        pages.append({"url": current_url, "title": title, "markdown": content})
                        logger.info("Playwright: Extracted %d chars from %s", len(content), current_url)

                    if current_depth < depth - 1 and len(pages) < breadth:
                        links = await page.evaluate(_EXTRACT_LINKS_JS)
                        for link in links:
                            link_parsed = urlparse(link)
                            if link_parsed.netloc == base_parsed.netloc:
                                normalized_link = f"{link_parsed.scheme}://{link_parsed.netloc}{link_parsed.path}"
                                if normalized_link not in visited:
                                    to_visit.append((link, current_depth + 1))

                except Exception as page_exc:
                    logger.warning("Playwright: Error crawling %s: %s", current_url, page_exc)
                    continue

            logger.info("Playwright crawl complete: %d pages extracted", len(pages))
            return pages
        finally:
            await browser.close()


def playwright_crawl(start_url: str, depth: int, breadth: int) -> list[dict]:
    """Crawl pages using Playwright (sync wrapper)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_playwright_crawl_async(start_url, depth, breadth))
    finally:
        loop.close()


async def _puppeteer_crawl_async(start_url: str, depth: int, breadth: int) -> list[dict]:
    """Crawl pages using Puppeteer (async implementation).

    Args:
        start_url: The URL to start crawling from
        depth: How deep to follow links (1 = only start page)
        breadth: Maximum number of pages to crawl

    Returns:
        List of page dicts with 'url', 'title', and 'markdown' keys
    """
    logger.info("Puppeteer crawl starting: url=%s, depth=%d, breadth=%d", start_url, depth, breadth)

    visited = set()
    pages = []
    to_visit = [(start_url, 0)]  # (url, current_depth)

    browser = None
    try:
        browser = await pyppeteer.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        page = await browser.newPage()
        await page.setUserAgent(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        while to_visit and len(pages) < breadth:
            current_url, current_depth = to_visit.pop(0)

            # Normalize URL and skip if already visited
            parsed = urlparse(current_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if normalized_url in visited:
                continue
            visited.add(normalized_url)

            logger.info("Puppeteer crawling: %s (depth=%d)", current_url, current_depth)

            try:
                response = await page.goto(current_url, {
                    'waitUntil': 'networkidle2',
                    'timeout': 30000
                })

                if not response or response.status >= 400:
                    logger.warning("Puppeteer: Failed to load %s (status=%s)",
                                   current_url, response.status if response else 'no response')
                    continue

                # Get page title
                title = await page.title() or ""

                # Extract text content from body
                content = await page.evaluate(_EXTRACT_TEXT_JS)

                if content:
                    pages.append({
                        'url': current_url,
                        'title': title,
                        'markdown': content
                    })
                    logger.info("Puppeteer: Extracted %d chars from %s", len(content), current_url)

                # If we haven't reached max depth, extract links to follow
                if current_depth < depth - 1 and len(pages) < breadth:
                    links = await page.evaluate(_EXTRACT_LINKS_JS)

                    # Filter to same domain and add to queue
                    base_parsed = urlparse(start_url)
                    for link in links:
                        link_parsed = urlparse(link)
                        if link_parsed.netloc == base_parsed.netloc:
                            normalized_link = f"{link_parsed.scheme}://{link_parsed.netloc}{link_parsed.path}"
                            if normalized_link not in visited:
                                to_visit.append((link, current_depth + 1))

            except Exception as page_exc:
                logger.warning("Puppeteer: Error crawling %s: %s", current_url, page_exc)
                continue

        logger.info("Puppeteer crawl complete: %d pages extracted", len(pages))
        return pages

    finally:
        if browser:
            await browser.close()


def puppeteer_crawl(start_url: str, depth: int, breadth: int) -> list[dict]:
    """Crawl pages using Puppeteer (sync wrapper).

    Args:
        start_url: The URL to start crawling from
        depth: How deep to follow links (1 = only start page)
        breadth: Maximum number of pages to crawl

    Returns:
        List of page dicts with 'url', 'title', and 'markdown' keys
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_puppeteer_crawl_async(start_url, depth, breadth))
    finally:
        loop.close()


def selenium_crawl(start_url: str, depth: int, breadth: int) -> list[dict]:
    """Crawl pages using Selenium WebDriver.

    Args:
        start_url: The URL to start crawling from
        depth: How deep to follow links (1 = only start page)
        breadth: Maximum number of pages to crawl

    Returns:
        List of page dicts with 'url', 'title', and 'markdown' keys
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    logger.info("Selenium crawl starting: url=%s, depth=%d, breadth=%d", start_url, depth, breadth)

    visited = set()
    pages = []
    to_visit = [(start_url, 0)]
    base_parsed = urlparse(start_url)

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(options=options)

        while to_visit and len(pages) < breadth:
            current_url, current_depth = to_visit.pop(0)

            parsed = urlparse(current_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if normalized_url in visited:
                continue
            visited.add(normalized_url)

            logger.info("Selenium crawling: %s (depth=%d)", current_url, current_depth)

            try:
                driver.get(current_url)

                # Check status via JavaScript (Selenium doesn't expose response status directly)
                status = driver.execute_script(
                    "return window.performance && window.performance.getEntriesByType('navigation')[0] "
                    "? window.performance.getEntriesByType('navigation')[0].responseStatus : 200"
                )
                if status and status >= 400:
                    logger.warning("Selenium: Failed to load %s (status=%s)", current_url, status)
                    continue

                title = driver.title or ""
                content = driver.execute_script(f"return ({_EXTRACT_TEXT_JS})()")

                if content:
                    pages.append({"url": current_url, "title": title, "markdown": content})
                    logger.info("Selenium: Extracted %d chars from %s", len(content), current_url)

                if current_depth < depth - 1 and len(pages) < breadth:
                    links = driver.execute_script(f"return ({_EXTRACT_LINKS_JS})()")
                    for link in links:
                        link_parsed = urlparse(link)
                        if link_parsed.netloc == base_parsed.netloc:
                            normalized_link = f"{link_parsed.scheme}://{link_parsed.netloc}{link_parsed.path}"
                            if normalized_link not in visited:
                                to_visit.append((link, current_depth + 1))

            except Exception as page_exc:
                logger.warning("Selenium: Error crawling %s: %s", current_url, page_exc)
                continue

        logger.info("Selenium crawl complete: %d pages extracted", len(pages))
        return pages
    finally:
        if driver:
            driver.quit()


def download_pdf(url: str) -> bytes:
    """Download a PDF file from a URL."""
    logger.info("Downloading PDF from: %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    content_length = len(response.content)
    logger.info("Downloaded PDF: %d bytes", content_length)
    return response.content


def extract_text_from_pdf(pdf_content: bytes) -> list[dict]:
    """
    Extract text from PDF content.
    Returns a list of page dictionaries with 'page_number' and 'text' keys.
    """
    logger.info("Extracting text from PDF (%d bytes)", len(pdf_content))
    pages = []

    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        logger.info("PDF has %d pages", total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "page_number": page_num,
                    "text": text,
                })
                logger.info("Extracted %d characters from page %d", len(text), page_num)
            else:
                logger.warning("Page %d has no extractable text", page_num)

        logger.info("Successfully extracted text from %d/%d pages", len(pages), total_pages)
    except Exception as exc:
        logger.error("Failed to extract text from PDF: %s", exc)
        raise

    return pages


def combine_pdf_pages(pages: list[dict], source_url: str) -> str:
    """Combine PDF page texts into a single document string."""
    combined = []
    for page in pages:
        page_num = page.get("page_number", "?")
        text = page.get("text", "")
        if text:
            header = f"Page {page_num} | Source: {source_url}"
            combined.append(f"{header}\n{text}")
    return "\n\n".join(combined).strip()


def chunk_text_by_character(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks by character count."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_text_by_sentence(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks by sentence boundaries."""
    import re
    if not text:
        return []
    
    # Split text into sentences (handles common sentence endings)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we have content, save current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text_content = ' '.join(current_chunk)
            chunks.append(chunk_text_content)
            
            # Calculate overlap: keep sentences from the end that fit within overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1  # +1 for space
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1 if current_chunk else 0
        
        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space
    
    # Add the last chunk
    if current_chunk:
        chunk_text_content = ' '.join(current_chunk)
        chunks.append(chunk_text_content)
    
    return chunks


def chunk_text_by_paragraph(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks by paragraph boundaries."""
    if not text:
        return []
    
    # Split text into paragraphs (double newlines or more)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        para_len = len(paragraph)
        
        # If adding this paragraph exceeds chunk_size and we have content, save current chunk
        if current_length + para_len > chunk_size and current_chunk:
            chunk_text_content = '\n\n'.join(current_chunk)
            chunks.append(chunk_text_content)
            
            # Calculate overlap: keep paragraphs from the end that fit within overlap
            overlap_paragraphs = []
            overlap_length = 0
            for p in reversed(current_chunk):
                if overlap_length + len(p) <= overlap:
                    overlap_paragraphs.insert(0, p)
                    overlap_length += len(p) + 2  # +2 for \n\n
                else:
                    break
            
            current_chunk = overlap_paragraphs
            current_length = sum(len(p) for p in current_chunk) + (len(current_chunk) - 1) * 2 if current_chunk else 0
        
        current_chunk.append(paragraph)
        current_length += para_len + 2  # +2 for \n\n
    
    # Add the last chunk
    if current_chunk:
        chunk_text_content = '\n\n'.join(current_chunk)
        chunks.append(chunk_text_content)
    
    return chunks


def chunk_text_by_semantic(text, chunk_size=1200, overlap=200, model=None):
    """
    Split text into chunks based on semantic similarity between sentences.
    
    This strategy uses embeddings to find natural semantic boundaries in the text,
    grouping semantically related sentences together while respecting chunk size limits.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (default: 1200)
        overlap: Number of sentences to overlap between chunks (default: 200, interpreted as ~1-2 sentences)
        model: SentenceTransformer model for generating embeddings (required)
    
    Returns:
        List of text chunks
    """
    import re
    if not text:
        return []
    
    if model is None:
        logger.warning("Semantic splitting requires a model, falling back to sentence splitting")
        return chunk_text_by_sentence(text, chunk_size, overlap)
    
    # Split text into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return sentences
    
    # Generate embeddings for all sentences
    logger.info("Generating embeddings for %d sentences for semantic chunking", len(sentences))
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    
    # Calculate similarity between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)
    
    # Find semantic breakpoints where similarity drops significantly
    # Use adaptive threshold based on mean and standard deviation
    if similarities:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        # Breakpoint threshold: below mean - 0.5 * std indicates a semantic boundary
        threshold = mean_sim - 0.5 * std_sim
        logger.info("Semantic chunking: mean_sim=%.3f, std_sim=%.3f, threshold=%.3f", 
                   mean_sim, std_sim, threshold)
    else:
        threshold = 0.5
    
    # Group sentences into chunks based on semantic boundaries and size constraints
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        
        # Check if we should start a new chunk
        should_break = False
        
        # Size constraint: if adding this sentence exceeds chunk_size
        if current_length + sentence_len > chunk_size and current_chunk:
            should_break = True
        # Semantic boundary: if similarity with previous sentence is below threshold
        elif i > 0 and i - 1 < len(similarities) and similarities[i - 1] < threshold:
            # Only break if current chunk has reasonable size (at least 20% of chunk_size)
            if current_length > chunk_size * 0.2:
                should_break = True
        
        if should_break:
            chunk_text_content = ' '.join(current_chunk)
            chunks.append(chunk_text_content)
            
            # Calculate overlap: keep last few sentences that fit within overlap chars
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1 if current_chunk else 0
        
        current_chunk.append(sentence)
        current_length += sentence_len + 1
    
    # Add the last chunk
    if current_chunk:
        chunk_text_content = ' '.join(current_chunk)
        chunks.append(chunk_text_content)
    
    logger.info("Semantic chunking created %d chunks from %d sentences", len(chunks), len(sentences))
    return chunks


# Supported splitting strategies (semantic handled separately due to model requirement)
SPLITTING_STRATEGIES = {
    "character": chunk_text_by_character,
    "sentence": chunk_text_by_sentence,
    "paragraph": chunk_text_by_paragraph,
    "semantic": chunk_text_by_semantic,
}

# Default values for indexing parameters
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SPLITTING_STRATEGY = "character"


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP, strategy=DEFAULT_SPLITTING_STRATEGY, model=None):
    """
    Split text into overlapping chunks using the specified strategy.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (default: 1200)
        overlap: Number of characters/content to overlap between chunks (default: 200)
        strategy: Splitting strategy - 'character', 'sentence', 'paragraph', or 'semantic' (default: 'character')
        model: SentenceTransformer model (required for 'semantic' strategy)
    
    Returns:
        List of text chunks
    """
    if strategy not in SPLITTING_STRATEGIES:
        logger.warning("Unknown splitting strategy '%s', falling back to 'character'", strategy)
        strategy = "character"
    
    chunk_func = SPLITTING_STRATEGIES[strategy]
    
    # Semantic strategy requires the model parameter
    if strategy == "semantic":
        return chunk_func(text, chunk_size, overlap, model=model)
    
    return chunk_func(text, chunk_size, overlap)


def cosine_similarity(query_embedding, chunk_embeddings):
    """Compute cosine similarity between query and chunk embeddings."""
    if not chunk_embeddings:
        return np.array([])
    chunk_matrix = np.vstack(chunk_embeddings)
    return chunk_matrix @ query_embedding


def index_document_chunks(
    document_id: str,
    text: str,
    database_name: str,
    collection_name: str,
    index_database_name: str = None,
    index_collection_name: str = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    splitting_strategy: str = DEFAULT_SPLITTING_STRATEGY,
):
    """
    Create vector-indexed chunks for a document.

    Args:
        document_id: The document identifier
        text: The text content to index
        database_name: The source database name (used for metadata updates)
        collection_name: The source collection name (used for metadata updates)
        index_database_name: The database where chunks will be stored (defaults to database_name)
        index_collection_name: The collection name for chunks (defaults to collection_name)
        chunk_size: Maximum size of each chunk (default: 1200)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)
        splitting_strategy: Strategy for splitting text - 'character', 'sentence', 'paragraph', or 'semantic' (default: 'character')

    Returns dict with indexing results or None if embedding model not configured.
    """
    # Use source values as defaults if index values not provided
    if index_database_name is None:
        index_database_name = database_name
    if index_collection_name is None:
        index_collection_name = collection_name

    logger.info("Starting indexing for document: %s", document_id)
    logger.info("Source: %s.%s, Index target: %s.%s",
                database_name, collection_name, index_database_name, index_collection_name)
    logger.info("Indexing parameters: chunk_size=%d, chunk_overlap=%d, splitting_strategy=%s",
                chunk_size, chunk_overlap, splitting_strategy)

    # Look up embedding model: use app default for privacy-compliance, else per-database
    if index_database_name == PRIVACY_COMPLIANCE_DB:
        model_name = get_application_embedding_model() or get_embedding_model_name(index_database_name)
    else:
        model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.info("Skipping indexing - no embedding model configured for database: %s", index_database_name)
        return None

    # Load the embedding model (needed for both semantic chunking and final embeddings)
    logger.info("Loading embedding model: %s", model_name)
    model = get_model(model_name)

    logger.info("Chunking document text (length: %d characters)", len(text))
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap, strategy=splitting_strategy, model=model)
    if not chunks:
        logger.warning("Document %s has no content to chunk", document_id)
        return None

    logger.info("Created %d chunks from document %s", len(chunks), document_id)

    logger.info("Generating embeddings for %d chunks", len(chunks))
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    logger.info("Embeddings generated successfully (dimension: %d)", embeddings.shape[1])

    # Write chunks to the index database/collection (use collection name verbatim)
    index_db = mongo_client[index_database_name]
    chunk_collection = index_collection_name

    logger.info("Clearing existing chunks for document %s in %s.%s",
                document_id, index_database_name, chunk_collection)
    delete_result = index_db[chunk_collection].delete_many({"document_id": document_id})
    logger.info("Deleted %d existing chunks", delete_result.deleted_count)

    chunk_docs = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_docs.append(
            {
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding.tolist(),
                "created_at": utc_now(),
            }
        )

    logger.info("Inserting %d chunks into %s.%s", len(chunk_docs), index_database_name, chunk_collection)
    index_db[chunk_collection].insert_many(chunk_docs)

    wg_db = mongo_client[WEB_GATHER_DB]
    wg_db[DOCUMENTS_COLLECTION].update_one(
        {"document_id": document_id},
        {
            "$set": {
                "indexed_at": utc_now(),
                "chunk_count": len(chunk_docs),
                "embedding_model": model_name,
                "chunk_collection": chunk_collection,
                "index_database_name": index_database_name,
                "index_collection_name": index_collection_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "splitting_strategy": splitting_strategy,
            }
        },
    )

    logger.info("Indexing complete for document %s: %d chunks indexed with model %s to %s.%s",
                document_id, len(chunk_docs), model_name, index_database_name, chunk_collection)

    return {
        "chunks_indexed": len(chunk_docs),
        "embedding_model": model_name,
        "chunk_collection": chunk_collection,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "splitting_strategy": splitting_strategy,
    }


def serialize_search_result(result, query=None):
    """Convert Firecrawl search result object to dict with relevance scoring.

    Args:
        result: Firecrawl search result (object or dict)
        query: Optional search query for calculating relevance score
    """
    if isinstance(result, dict):
        data = result.copy()
    else:
        # Handle SearchResultWeb or Document objects
        data = {}
        if hasattr(result, "url"):
            data["url"] = result.url
        if hasattr(result, "title"):
            data["title"] = result.title
        if hasattr(result, "description"):
            data["description"] = result.description
        if hasattr(result, "markdown"):
            data["markdown"] = result.markdown
        if hasattr(result, "metadata") and result.metadata:
            data["url"] = getattr(result.metadata, "url", None) or data.get("url")
            data["title"] = getattr(result.metadata, "title", None) or data.get("title")

    # Calculate relevance score based on query matching
    if query:
        score = calculate_relevance_score(
            query,
            data.get("title", ""),
            data.get("description", "")
        )
        data["score"] = score
        data["percent_match"] = round(score * 100, 1)

    return data


@core_bp.post("/gather")
def gather():
    """Gather web documents based on query."""
    logger.info("POST /gather - Starting web search")
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    logger.info("POST /gather - Parameters: query=%s", query)
    if not query:
        logger.warning("POST /gather - Missing required parameter: query")
        return jsonify({"error": "query is required"}), 400

    logger.info("POST /gather - Searching for: %s", query)
    results = firecrawl_client.search(
        query=query,
        limit=10,
    )
    normalized = normalize_results(results)
    serialized = [serialize_search_result(r, query=query) for r in normalized]
    logger.info("POST /gather - Found %d results for query: %s", len(serialized), query)

    # Build next_step information for the gather->ingest flow
    next_step = {
        "action": "Select a URL to crawl and call POST /ingest",
        "endpoint": "/ingest",
        "method": "POST",
        "required_parameters": {
            "url": {
                "type": "string",
                "description": "Select a URL from the results above to crawl",
            },
            "database": {
                "type": "string",
                "description": "Database name for storing raw crawled data",
            },
            "collection": {
                "type": "string",
                "description": "Collection name for storing raw crawled data",
            },
        },
        "optional_parameters": {
            "depth": {
                "type": "integer",
                "default": 1,
                "description": "Crawl depth for web pages",
            },
            "breadth": {
                "type": "integer",
                "default": 5,
                "description": "Max pages to crawl for web pages",
            },
            "mode": {
                "type": "string",
                "enum": ["append", "overwrite"],
                "default": "append",
                "description": "How to handle existing data: 'append' adds to existing data, 'overwrite' clears existing data first",
            },
            "index_database": {
                "type": "string",
                "description": "Database name for storing chunked/vectorized data (enables automatic indexing)",
            },
            "index_collection": {
                "type": "string",
                "description": "Collection name for storing chunked/vectorized data (enables automatic indexing)",
            },
        },
    }

    return jsonify({
        "query": query,
        "results": serialized,
        "next_step": next_step,
    })


@core_bp.post("/ingest")
def ingest():
    """Load a document by crawling a URL or parsing a PDF."""
    logger.info("POST /ingest - Starting document ingestion (load only)")
    payload = request.get_json(silent=True) or {}
    url = payload.get("url")
    depth = int(payload.get("depth", 1))
    breadth = int(payload.get("breadth", 5))
    database_name = payload.get("database")
    collection_name = payload.get("collection")
    mode = payload.get("mode", "append").lower()
    logger.info("POST /ingest - Parameters: url=%s, depth=%d, breadth=%d, database=%s, collection=%s, mode=%s",
                url, depth, breadth, database_name, collection_name, mode)

    if not url or not database_name or not collection_name:
        logger.warning("POST /ingest - Missing required parameters")
        return (
            jsonify(
                {
                    "error": "url, database, and collection are required",
                }
            ),
            400,
        )

    # Validate mode parameter
    if mode not in ("append", "overwrite"):
        logger.warning("POST /ingest - Invalid mode: %s", mode)
        return (
            jsonify(
                {
                    "error": "mode must be 'append' or 'overwrite'",
                }
            ),
            400,
        )

    # Detect if URL is a PDF file
    is_pdf = is_pdf_url(url)
    if not is_pdf:
        # Check content type via HEAD request
        content_type = detect_content_type(url)
        is_pdf = "application/pdf" in content_type

    page_count = 0
    document_type = "web"

    if is_pdf:
        # Handle PDF file
        logger.info("POST /ingest - Detected PDF file, downloading and extracting text")
        document_type = "pdf"
        try:
            pdf_content = download_pdf(url)
            pdf_pages = extract_text_from_pdf(pdf_content)
            page_count = len(pdf_pages)

            if not pdf_pages:
                logger.warning("POST /ingest - PDF has no extractable text: %s", url)
                return jsonify({"error": "PDF has no extractable text"}), 400

            combined_text = combine_pdf_pages(pdf_pages, url)
            logger.info("POST /ingest - Extracted text from %d PDF pages", page_count)

        except requests.RequestException as exc:
            logger.error("POST /ingest - Failed to download PDF %s: %s", url, exc)
            return jsonify({"error": f"Failed to download PDF: {exc}"}), 500
        except Exception as exc:
            logger.error("POST /ingest - Failed to parse PDF %s: %s", url, exc)
            return jsonify({"error": f"Failed to parse PDF: {exc}"}), 500
    else:
        # Handle web page via Firecrawl
        logger.info("POST /ingest - Crawling web URL: %s (depth=%d, breadth=%d)", url, depth, breadth)
        try:
            crawl_result = firecrawl_client.crawl(
                url=url,
                limit=breadth,
            )
        except Exception as exc:
            logger.error("POST /ingest - Crawl failed for URL %s: %s", url, exc)
            return jsonify({"error": f"crawl failed: {exc}"}), 500

        # Debug: log crawl result structure
        logger.info("POST /ingest - crawl_result type: %s", type(crawl_result))
        if hasattr(crawl_result, '__dict__'):
            logger.info("POST /ingest - crawl_result attrs: %s", list(vars(crawl_result).keys()))
        if hasattr(crawl_result, 'data'):
            logger.info("POST /ingest - crawl_result.data: type=%s, len=%s",
                       type(crawl_result.data), len(crawl_result.data) if crawl_result.data else 0)
        if hasattr(crawl_result, 'status'):
            logger.info("POST /ingest - crawl_result.status: %s", crawl_result.status)

        pages = normalize_results(crawl_result)
        page_count = len(pages)
        logger.info("POST /ingest - Crawl returned %d pages", page_count)

        # Check if crawl returned no pages at all
        if page_count == 0:
            logger.error("POST /ingest - Crawl returned 0 pages for URL: %s", url)
            return jsonify({"error": "crawl returned no pages - website may be blocking crawlers"}), 400

        # Log any warnings from crawled pages
        for idx, page in enumerate(pages):
            if hasattr(page, 'warning') and page.warning:
                logger.warning("POST /ingest - Page %d warning: %s", idx, page.warning)
            if hasattr(page, 'metadata') and page.metadata:
                status_code = getattr(page.metadata, 'statusCode', None)
                if status_code and status_code != 200:
                    logger.warning("POST /ingest - Page %d statusCode: %s", idx, status_code)

        combined_text = combine_pages(pages)
        if not combined_text:
            logger.warning("POST /ingest - Crawl returned no content for URL: %s", url)
            return jsonify({"error": "crawl returned no content"}), 400

    logger.info("POST /ingest - Combined text length: %d characters", len(combined_text))

    db = mongo_client[database_name]
    wg_db = mongo_client[WEB_GATHER_DB]

    # Track previous document count for response
    previous_document_count = 0
    overwritten = False

    # Handle overwrite mode - clear existing data in collection
    if mode == "overwrite":
        logger.info("POST /ingest - Overwrite mode: checking for existing data in %s.%s", database_name, collection_name)
        previous_document_count = db[collection_name].count_documents({})

        if previous_document_count > 0:
            logger.info("POST /ingest - Overwrite mode: clearing %d existing documents", previous_document_count)

            # Delete documents from the main collection
            db[collection_name].delete_many({})

            # Delete document metadata records for this collection
            wg_db[DOCUMENTS_COLLECTION].delete_many({
                "database_name": database_name,
                "collection_name": collection_name,
            })

            overwritten = True
            logger.info("POST /ingest - Overwrite mode: existing data cleared successfully")

    document_id = str(uuid.uuid4())
    logger.info("POST /ingest - Created document ID: %s", document_id)

    logger.info("POST /ingest - Storing document in %s.%s (mode: %s)", database_name, collection_name, mode)
    db[collection_name].insert_one(
        {
            "_id": document_id,
            "source_url": url,
            "document_type": document_type,
            "text": combined_text,
            "created_at": utc_now(),
        }
    )

    description = combined_text[:300]
    logger.info("POST /ingest - Creating document metadata record")
    wg_db[DOCUMENTS_COLLECTION].insert_one(
        {
            "document_id": document_id,
            "database_name": database_name,
            "collection_name": collection_name,
            "source_url": url,
            "document_type": document_type,
            "description": description,
            "created_at": utc_now(),
        }
    )

    response_data = {
        "document_id": document_id,
        "document_type": document_type,
        "pages": page_count,
        "database_name": database_name,
        "collection_name": collection_name,
        "mode": mode,
        "message": (
            "Document loaded successfully. Use /create-embeddings endpoint to create "
            "vector embeddings."
        ),
    }

    if mode == "overwrite":
        response_data["overwritten"] = overwritten
        response_data["previous_document_count"] = previous_document_count

    logger.info("POST /ingest - Document loaded successfully (mode: %s): %s", mode, document_id)
    return jsonify(response_data)


@core_bp.post("/create-embeddings")
def index_document():
    """Index collection rows by embedding text from the specified column."""
    logger.info("POST /create-embeddings - Starting document indexing")
    payload = request.get_json(silent=True) or {}

    # Get required parameters
    source_database_name = payload.get("source_database_name")
    source_collection_name = payload.get("source_collection_name")
    index_database_name = payload.get("index_database_name")
    index_collection_name = payload.get("index_collection_name")
    source_query_param = payload.get("source_query")
    text_column = payload.get("text_column") or "chunk_text"

    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str):
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "POST /create-embeddings - Invalid JSON in source_query: %s",
                    str(exc),
                )
                return (
                    jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}),
                    400,
                )
        else:
            logger.warning("POST /create-embeddings - source_query must be a JSON object")
            return jsonify({"error": "source_query must be a JSON object"}), 400

        if not isinstance(source_query, dict):
            logger.warning("POST /create-embeddings - source_query must be a JSON object")
            return jsonify({"error": "source_query must be a JSON object"}), 400

    logger.info(
        "POST /create-embeddings - Parameters: source_database_name=%s, "
        "source_collection_name=%s, index_database_name=%s, "
        "index_collection_name=%s, text_column=%s",
        source_database_name,
        source_collection_name,
        index_database_name,
        index_collection_name,
        text_column,
    )
    if source_query:
        logger.info(
            "POST /create-embeddings - source_query for %s.%s: %s",
            source_database_name,
            source_collection_name,
            source_query,
        )

    # Validate required parameters
    missing_params = []
    if not source_database_name:
        missing_params.append("source_database_name")
    if not source_collection_name:
        missing_params.append("source_collection_name")
    if not index_database_name:
        missing_params.append("index_database_name")
    if not index_collection_name:
        missing_params.append("index_collection_name")

    if missing_params:
        logger.warning(
            "POST /create-embeddings - Missing required parameters: %s",
            ", ".join(missing_params),
        )
        return jsonify({
            "error": f"Missing required parameters: {', '.join(missing_params)}"
        }), 400

    if not isinstance(text_column, str) or not text_column.strip():
        return jsonify({"error": "text_column must be a non-empty string"}), 400
    text_column = text_column.strip()

    logger.info(
        "POST /create-embeddings - Source: %s.%s",
        source_database_name,
        source_collection_name,
    )
    logger.info(
        "POST /create-embeddings - Index target: %s.%s",
        index_database_name,
        index_collection_name,
    )

    # Use app default for privacy-compliance, else per-database embedding model
    if index_database_name == PRIVACY_COMPLIANCE_DB:
        model_name = get_application_embedding_model() or get_embedding_model_name(index_database_name)
    else:
        model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.error(
            "POST /create-embeddings - No embedding model configured for database: %s",
            index_database_name,
        )
        return jsonify({
            "error": f"No embedding model configured for index_database_name '{index_database_name}'. "
                     f"Use POST /embedding-models to configure one."
        }), 400

    source_db = mongo_client[source_database_name]
    source_docs = list(source_db[source_collection_name].find(source_query))
    if not source_docs:
        logger.warning(
            "POST /create-embeddings - No documents found in %s.%s",
            source_database_name,
            source_collection_name,
        )
        return jsonify({
            "error": f"No documents found in {source_database_name}.{source_collection_name}"
        }), 404

    docs_to_index = []
    chunk_texts = []
    skipped_rows = 0
    for doc in source_docs:
        raw_text = doc.get(text_column)
        text = str(raw_text) if raw_text is not None else ""
        if not text.strip():
            skipped_rows += 1
            continue
        docs_to_index.append(doc)
        chunk_texts.append(text)

    if not docs_to_index:
        logger.warning(
            "POST /create-embeddings - No rows with %s found in %s.%s",
            text_column,
            source_database_name,
            source_collection_name,
        )
        return jsonify({"error": f"no rows with '{text_column}' to index"}), 400

    logger.info("POST /create-embeddings - Loading embedding model: %s", model_name)
    model = get_model(model_name)

    logger.info(
        "POST /create-embeddings - Generating embeddings for %d rows",
        len(chunk_texts),
    )
    embeddings = model.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True
    )

    index_db = mongo_client[index_database_name]
    index_collection = index_db[index_collection_name]
    same_target = (
        source_database_name == index_database_name
        and source_collection_name == index_collection_name
    )

    if same_target:
        for doc, embedding, chunk_text in zip(docs_to_index, embeddings, chunk_texts):
            set_fields = {
                "embedding": embedding.tolist(),
                text_column: chunk_text,  # Embedded text goes to text_column (e.g. subchunk_text)
                "indexed_at": utc_now(),
                "source_id": str(doc.get("_id")),
                "source_database_name": source_database_name,
                "source_collection_name": source_collection_name,
            }
            if doc.get("category") is not None:
                set_fields["category"] = doc["category"]
            index_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": set_fields},
            )
    else:
        delete_filter = {
            "source_database_name": source_database_name,
            "source_collection_name": source_collection_name,
        }
        if source_query:
            delete_filter.update(source_query)
        deleted = index_collection.delete_many(delete_filter)
        if deleted.deleted_count:
            logger.info(
                "POST /create-embeddings - Removed %d existing index rows matching source_query in %s.%s",
                deleted.deleted_count,
                index_database_name,
                index_collection_name,
            )
        index_docs = []
        for doc, embedding, chunk_text in zip(docs_to_index, embeddings, chunk_texts):
            base_doc = {key: value for key, value in doc.items() if key != "_id"}
            index_doc = {
                **base_doc,
                "source_id": str(doc.get("_id")),
                text_column: chunk_text,  # Embedded text to text_column; chunk_text preserved from base_doc
                "embedding": embedding.tolist(),
                "indexed_at": utc_now(),
                "source_database_name": source_database_name,
                "source_collection_name": source_collection_name,
            }
            if doc.get("category") is not None:
                index_doc["category"] = doc["category"]
            index_docs.append(index_doc)
        logger.info(
            "POST /create-embeddings - Inserting %d rows into %s.%s",
            len(index_docs),
            index_database_name,
            index_collection_name,
        )
        index_collection.insert_many(index_docs)

    logger.info(
        "POST /create-embeddings - Successfully indexed %d rows",
        len(docs_to_index),
    )
    return jsonify(
        {
            "source_database_name": source_database_name,
            "source_collection_name": source_collection_name,
            "index_database_name": index_database_name,
            "index_collection_name": index_collection_name,
            "text_column": text_column,
            "chunks_indexed": len(docs_to_index),
            "skipped_rows": skipped_rows,
            "embedding_model": model_name,
        }
    )


@core_bp.get("/search")
def search():
    """Search vector-indexed collection."""
    logger.info("GET /search - Starting vector search")
    document_id = request.args.get("document_id")
    query = request.args.get("query")
    logger.info("GET /search - Parameters: document_id=%s, query=%s", document_id, query)
    if not document_id or not query:
        logger.warning("GET /search - Missing required parameters")
        return jsonify({"error": "document_id and query are required"}), 400

    logger.info("GET /search - Searching document %s for: %s", document_id, query)
    wg_db = mongo_client[WEB_GATHER_DB]
    doc_record = wg_db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id})
    if not doc_record:
        logger.warning("GET /search - Document not found: %s", document_id)
        return jsonify({"error": "document_id not found"}), 404

    database_name = doc_record["database_name"]
    collection_name = doc_record["collection_name"]

    # Use index location from metadata if available, otherwise fall back to source location
    index_database_name = doc_record.get("index_database_name", database_name)
    chunk_collection = doc_record.get("chunk_collection", collection_name)

    if index_database_name == PRIVACY_COMPLIANCE_DB:
        model_name = get_application_embedding_model() or get_embedding_model_name(index_database_name)
    else:
        model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.error("GET /search - No embedding model configured for database: %s", index_database_name)
        return jsonify({"error": "embedding model not configured"}), 400

    db = mongo_client[index_database_name]
    logger.info("GET /search - Loading chunks from %s.%s", index_database_name, chunk_collection)
    chunks = list(
        db[chunk_collection].find({"document_id": document_id}, {"_id": 0})
    )
    if not chunks:
        logger.warning("GET /search - No chunks indexed for document: %s", document_id)
        return jsonify({"error": "no chunks indexed for document"}), 400

    logger.info("GET /search - Found %d chunks, generating query embedding", len(chunks))
    model = get_model(model_name)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    chunk_embeddings = [np.array(c["embedding"]) for c in chunks]

    logger.info("GET /search - Computing cosine similarity scores")
    scores = cosine_similarity(query_embedding, chunk_embeddings)

    results = []
    for chunk, score in zip(chunks, scores):
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "percent_match": max(0.0, min(100.0, float(score) * 100.0)),
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)

    top_results = results[:5]
    logger.info("GET /search - Returning top %d results (best score: %.3f)",
                len(top_results), top_results[0]["score"] if top_results else 0)
    return jsonify(
        {
            "document_id": document_id,
            "query": query,
            "results": top_results,
        }
    )


@core_bp.route("/vector-search", methods=["GET", "POST"])
def vector_search():
    """Vector search over a MongoDB Atlas vector index.

    Converts the query to embeddings and runs $vectorSearch on the given
    database, collection, and index.

    Parameters (query string or JSON body):
        database (str): Database name
        collection (str): Collection name
        index (str): Vector search index name
        query (str): Search query text (will be embedded). Omit if query_vector provided.
        query_vector (list): Precomputed vector for search. If provided, skips embedding.
        limit (int, optional): Max results to return (default 10)
        path (str, optional): Path to vector field in documents. If omitted,
            uses the first field path from the embedding_model for the database.
        filter (object, optional): MongoDB filter for $vectorSearch (e.g. {"jurisdiction": "CA"}).
            Can be a JSON object or JSON string. Model is from web-gather.embedding_model.
    """
    logger.info("%s /vector-search - Starting vector search", request.method)
    if request.method == "POST" and request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = dict(request.args)
    database = payload.get("database")
    collection = payload.get("collection")
    index = payload.get("index")
    query = payload.get("query")
    query_vector_param = payload.get("query_vector")
    limit = payload.get("limit", 10)
    path = payload.get("path")
    filter_param = payload.get("filter")

    missing = []
    if not database:
        missing.append("database")
    if not collection:
        missing.append("collection")
    if not index:
        missing.append("index")
    if not query and not query_vector_param:
        missing.append("query or query_vector")

    if missing:
        logger.warning("POST /vector-search - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    if query_vector_param is not None:
        if isinstance(query_vector_param, str):
            try:
                query_vector_param = json.loads(query_vector_param)
            except json.JSONDecodeError as exc:
                return jsonify({"error": f"query_vector must be valid JSON: {exc!s}"}), 400
        if not isinstance(query_vector_param, list):
            return jsonify({"error": "query_vector must be a list of numbers"}), 400
        try:
            query_vector = [float(x) for x in query_vector_param]
        except (TypeError, ValueError):
            return jsonify({"error": "query_vector must contain only numbers"}), 400
        if not query_vector:
            return jsonify({"error": "query_vector must not be empty"}), 400

    try:
        limit = int(limit) if limit is not None else 10
    except (TypeError, ValueError):
        limit = 10
    if limit < 1 or limit > 100:
        limit = 10

    filter_doc = None
    if filter_param is not None:
        if isinstance(filter_param, dict):
            filter_doc = filter_param
        elif isinstance(filter_param, str):
            try:
                filter_doc = json.loads(filter_param)
            except json.JSONDecodeError as exc:
                logger.warning("POST /vector-search - Invalid JSON in filter: %s", exc)
                return jsonify({"error": f"filter must be valid JSON: {exc!s}"}), 400
        else:
            return jsonify({"error": "filter must be a JSON object or JSON string"}), 400
        if not isinstance(filter_doc, dict):
            return jsonify({"error": "filter must be a JSON object"}), 400

    if query_vector_param is None:
        # Resolve embedding model and path from web-gather.embedding_model (same as /create-embeddings)
        if database == PRIVACY_COMPLIANCE_DB:
            model_name = get_application_embedding_model() or get_embedding_model_name(database)
        else:
            model_name = get_embedding_model_name(database)

        if not model_name:
            logger.warning("POST /vector-search - No embedding model for database: %s", database)
            return jsonify({
                "error": f"No embedding model configured for database '{database}'. "
                "Use POST /embedding-models to configure one."
            }), 400

        model = get_model(model_name)
        query_vector = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].tolist()

    if not path:
        record = get_embedding_model_record(database)
        if record and isinstance(record.get("fields"), list) and record["fields"]:
            path = record["fields"][0].get("path")
        if not path:
            logger.warning("POST /vector-search - Cannot determine vector path for database: %s", database)
            return jsonify({
                "error": "Vector field path unknown. Either configure embedding_model with 'fields' "
                "or pass 'path' in the request body."
            }), 400

    logger.info(
        "POST /vector-search - Querying %s.%s index=%s path=%s limit=%s",
        database, collection, index, path, limit,
    )

    vector_search_stage = {
        "index": index,
        "path": path,
        "queryVector": query_vector,
        "numCandidates": max(int(limit) * 10, 100),
        "limit": int(limit),
    }
    if filter_doc:
        vector_search_stage["filter"] = filter_doc

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    try:
        db = mongo_client[database]
        coll = db[collection]
        results = list(coll.aggregate(pipeline))
    except Exception as e:
        logger.exception("POST /vector-search - Aggregation failed")
        return jsonify({"error": f"Vector search failed: {e!s}"}), 500

    # Exclude raw vector from response (keep score from $meta)
    out = []
    for doc in results:
        d = dict(doc)
        if path in d:
            del d[path]
        if "_id" in d:
            d["_id"] = str(d["_id"])
        out.append(d)

    logger.info("POST /vector-search - Returning %d results", len(out))
    resp = {
        "database": database,
        "collection": collection,
        "index": index,
        "results": out,
    }
    if query is not None:
        resp["query"] = query
    return jsonify(resp)


def _split_into_paragraph_chunks(text: str) -> list[str]:
    """Split text into paragraph chunks (by double newlines)."""
    if not text:
        return []
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


@core_bp.post("/create-paragraph-sections")
def create_subsections():
    """Split source column into paragraph chunks and write to a new collection.

    For each record in source_collection, reads the value from `column`, splits
    it by paragraph boundaries (double newlines), and creates one record per
    subchunk in destination_collection. Each destination record includes all
    source columns except the split column, plus subsection_column (the subchunk
    text) and a unique subchunk_id (guid).

    Optional source_query: MongoDB query to filter source records (e.g. {"document_id": "x"}).
    When omitted, all records are processed.

    Example: database=privacy-compliance, source_collection=statute_chunks,
    destination_collection=statute_subchunks, column=chunk_text,
    subsection_column=subchunk_text, source_query={"jurisdiction": "California"}.
    """
    logger.info("POST /create-paragraph-sections - Starting")
    payload = request.get_json(silent=True) or {}
    database = payload.get("database")
    source_collection = payload.get("source_collection")
    destination_collection = payload.get("destination_collection")
    column = payload.get("column")
    subsection_column = payload.get("subsection_column")
    source_query_param = payload.get("source_query")
    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str) and source_query_param.strip():
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning("POST /create-paragraph-sections - Invalid JSON in source_query: %s", exc)
                return jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}), 400
        if not isinstance(source_query, dict):
            source_query = {}

    missing = []
    if not database:
        missing.append("database")
    if not source_collection:
        missing.append("source_collection")
    if not destination_collection:
        missing.append("destination_collection")
    if not column:
        missing.append("column")
    if not subsection_column:
        missing.append("subsection_column")

    if missing:
        logger.warning("POST /create-paragraph-sections - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    _warn_if_database_or_collection_not_found(
        database, source_collection, "POST /create-paragraph-sections",
    )

    logger.info(
        "POST /create-paragraph-sections - %s.%s -> %s.%s column=%s subsection_column=%s source_query=%s",
        database, source_collection, database, destination_collection,
        column, subsection_column, source_query,
    )

    try:
        db = mongo_client[database]
        source_coll = db[source_collection]
        dest_coll = db[destination_collection]
        docs = list(source_coll.find(source_query))
    except Exception as e:
        logger.exception("POST /create-paragraph-sections - Failed to read source collection")
        return jsonify({"error": f"Failed to read source collection: {e!s}"}), 500

    records_inserted = 0
    source_rows_processed = 0
    source_rows_skipped = 0

    for doc in docs:
        raw = doc.get(column)
        text = str(raw) if raw is not None else ""
        subchunks = _split_into_paragraph_chunks(text)
        if not subchunks:
            source_rows_skipped += 1
            continue

        # Base document: all columns except the split column and _id
        base = {k: v for k, v in doc.items() if k != column and k != "_id"}
        # Add source_id for traceability
        if "_id" in doc:
            base["source_id"] = str(doc["_id"])

        for subchunk_text in subchunks:
            subchunk_text = " ".join(subchunk_text.split())
            record = {
                **base,
                subsection_column: subchunk_text,
                "subchunk_id": str(uuid.uuid4()),
                column: text,  # Original full chunk text for context
            }
            try:
                dest_coll.insert_one(record)
                records_inserted += 1
            except Exception as e:
                logger.warning("POST /create-paragraph-sections - Failed to insert: %s", e)
                break
        source_rows_processed += 1

    logger.info(
        "POST /create-paragraph-sections - Inserted %d records, processed %d source rows, skipped %d",
        records_inserted, source_rows_processed, source_rows_skipped,
    )
    return jsonify({
        "database": database,
        "source_collection": source_collection,
        "destination_collection": destination_collection,
        "column": column,
        "subsection_column": subsection_column,
        "records_inserted": records_inserted,
        "source_rows_processed": source_rows_processed,
        "source_rows_skipped": source_rows_skipped,
    })

old = """
Statute SECTIONS are defined ONLY by lowercase letters in parentheses: (a), (b), (c), (d), etc. Do NOT split on numeric markers like (1), (2), (3), (8)—those are nested subsections within a parent section and must be kept together.

Example: Section (d) may contain (1) through (8) as nested items. Output ONE subsection for (d) that includes all of (1) through (8) as part of its text.
Instructions:
- Preserve the original statutory language verbatim. Do NOT summarize, paraphrase, or interpret.
- Split ONLY at alphabetic section markers: (a), (b), (c), (d), (e), etc.
- Each subsection must include the chunk header (e.g., "# 1798.105. Consumers' Right to Delete Personal Information") at the start, followed by the section content.
- Remove all linefeeds from the output text: use spaces instead of newlines. Output each subsection as a single continuous line.
- Maintain the original order of sections.
- If the text has no clear alphabetic section markers, return a single subsection with the full text and identifier "(0)" or "()".

"""

# Default prompt for statute subsection extraction when parse_prompt is blank.
# Sections are defined by alphabetic (a), (b), (c), (d) only. Numeric (1), (2), (8) are nested.
STATUTE_SUBSECTION_DEFAULT_PROMPT = """You are a legal document classifier specializing in statutory interpretation.

Task:
Parse the provided statute section into its sections and categorize each section.
Parse the document based on legal context. 

Possible categories:
Given a section of a state privacy statute,
classify it into exactly one of the following categories.

CATEGORIES:

- "definitions": Sections that define terms used throughout the statute. These
  establish the meaning of key concepts (e.g., "consumer," "personal data,"
  "sensitive data") but impose no obligations or rights.

- "applicability": Sections that define who the statute applies to, including
  threshold requirements (e.g., number of consumers, revenue thresholds),
  geographic scope, and entity exemptions. These determine whether an
  organization falls under the statute but impose no specific duties.

- "consumer_rights": Sections that establish rights consumers may exercise
  against controllers, such as the right to access, delete, correct, or port
  personal data, or the right to opt out of sale, targeted advertising, or
  profiling.

- "controller_duties": Sections that impose affirmative obligations on
  controllers, such as providing privacy notices, limiting data collection,
  conducting data protection assessments, establishing opt-out mechanisms,
  or obtaining consent for sensitive data processing.

- "processor_duties": Sections that impose obligations on processors, such as
  contractual requirements, duty to assist controllers, confidentiality
  obligations, or sub-processor management.

- "enforcement": Sections that establish enforcement mechanisms, including
  attorney general authority, civil penalties, cure periods, private right
  of action (or lack thereof), and consumer complaint procedures.

- "other": Sections that do not fit the above categories, such as severability
  clauses, effective dates, or legislative findings.



Instructions:
- The document has line numbers at the start of each line (e.g., "1: ...", "2: ..."). Use these to identify section boundaries.
- For each subsection, report start_line and end_line (1-indexed, inclusive) instead of copying the full text.
- Preserve the original statutory language by referencing line ranges; do NOT summarize or paraphrase.
- Maintain the original order of sections.
- If the text has no clear alphabetic section markers, return a single subsection with start_line=1 and end_line=<last line>.

Output Format:
Return ONLY valid JSON with this exact structure (no surrounding text). Use start_line and end_line to reference the text; do NOT include a "text" field:
{
  "subsections": [
    {
      "header_text": "Consumer Rights – Right to invoke consumer rights",
      "identifier": "(1)",
      "category": "consumer_rights",
      "category_reasoning": "This section establishes the right to submit requests.",
      "start_line": 1,
      "end_line": 12
    },
    {
      "header_text": "Controller obligations to comply with requests",
      "identifier": "(2)",
      "category": "controller_duties",
      "category_reasoning": "This section imposes duties on controllers.",
      "start_line": 13,
      "end_line": 28
    }
  ]
}"""

# Default prompt for statute sub-topic extraction when parse_prompt is blank.
# Identifies distinct compliance requirements (sub_topics) from statute sections.
STATUTE_SUBTOPIC_DEFAULT_PROMPT = """You are a legal document analyst. Given a statute section that has already
been classified into a primary category, identify the specific sub_topic
that captures the distinct regulatory requirement.

INSTRUCTIONS:

1. Each sub_topic should represent a single, testable compliance requirement.
   If a statute section contains multiple distinct requirements, return
   multiple sub_topics.

2. Sub_topic names should be:
   - Lowercase with underscores (snake_case)
   - Descriptive enough to distinguish from other sub_topics in the same
     category
   - Consistent across state statutes that impose similar requirements
     (e.g., Virginia's and Kentucky's right to delete should both produce
     "right_to_delete", not state-specific naming)

3. For each sub_topic, provide a requirement_summary that captures what a
   company must specifically do or provide to comply. This summary should
   be concrete enough to evaluate against a privacy policy.

4. Assign the policy_categories most likely to contain relevant language
   for this sub_topic.

Respond with valid JSON matching this schema:
{
  "sub_topics": [
    {
      "sub_topic": "<snake_case identifier>",
      "requirement_summary": "<one sentence: what must the company do>",
      "policy_categories": ["<primary>", "<secondary if applicable>"],
      "requires_consent": <true if the requirement involves obtaining consent>,
      "consumer_facing": <true if the requirement involves a disclosure or
                          mechanism visible to consumers>
    }
  ]
}

COMMON SUB_TOPICS BY CATEGORY (use these when applicable, create new ones
only when the requirement does not fit an existing sub_topic):

consumer_rights:
  - right_to_access: Consumer can request what personal data is held
  - right_to_delete: Consumer can request deletion of personal data
  - right_to_correct: Consumer can request correction of inaccurate data
  - right_to_portability: Consumer can obtain their data in a portable format
  - right_to_opt_out_sale: Consumer can opt out of sale of personal data
  - right_to_opt_out_targeted_ads: Consumer can opt out of targeted advertising
  - right_to_opt_out_profiling: Consumer can opt out of automated profiling
  - right_to_appeal: Consumer can appeal a denied rights request
  - right_to_nondiscrimination: Consumer cannot be penalized for exercising rights
  - rights_request_process: How consumers submit and controller responds to requests

controller_duties:
  - privacy_notice: Must provide a clear and accessible privacy notice
  - purpose_limitation: Must limit processing to disclosed purposes
  - data_minimization: Must limit collection to what is adequate and necessary
  - sensitive_data_consent: Must obtain opt-in consent for sensitive data
  - child_data_consent: Must obtain parental consent for known children
  - data_security: Must implement reasonable security practices
  - data_protection_assessment: Must conduct assessments for high-risk processing
  - opt_out_mechanism: Must provide universal opt-out recognition or mechanism
  - response_timeline: Must respond to consumer requests within statutory period
  - third_party_disclosure: Must disclose categories of third parties receiving data
  - retention_disclosure: Must disclose retention periods or criteria

processor_duties:
  - contractual_requirements: Must have binding contract with controller
  - duty_to_assist: Must assist controller in fulfilling consumer requests
  - confidentiality: Must ensure personnel are bound by confidentiality
  - sub_processor_management: Must obtain controller approval for sub-processors
  - data_return_delete: Must return or delete data at end of relationship

For categories "definitions" or "applicability" that do not impose testable compliance
requirements, return an empty sub_topics array: {"sub_topics": []}."""


@core_bp.post("/create-statute-subsections")
def create_statute_subsections():
    """Split statute section column into subsections using LLM and write to destination collection.

    For each record in source_collection, reads the value from `column`, uses an LLM
    to identify statute sections by alphabetic markers (a), (b), (c), (d) only. Numeric
    markers (1), (2), (8) are nested within a section and are kept together. Each
    subsection includes the chunk header and has all linefeeds removed. Creates one
    record per section in destination_collection. Each destination record includes all
    source columns except the split column, plus subsection_column (the subsection text),
    subsection_identifier (e.g., (a), (b)), and a unique subchunk_id (guid).

    Optional parse_prompt: when provided, appended as additional parsing instructions.
    When blank, uses the default prompt for statute subsection extraction.
    """
    logger.info("POST /create-statute-subsections - Starting")
    payload = request.get_json(silent=True) or {}
    database = payload.get("database")
    source_collection = payload.get("source_collection")
    destination_collection = payload.get("destination_collection")
    column = payload.get("column")
    subsection_column = payload.get("subsection_column")
    parse_prompt = payload.get("parse_prompt") or ""
    source_query_param = payload.get("source_query")
    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str) and source_query_param.strip():
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning("POST /create-statute-subsections - Invalid JSON in source_query: %s", exc)
                return jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}), 400
        if not isinstance(source_query, dict):
            source_query = {}

    missing = []
    if not database:
        missing.append("database")
    if not source_collection:
        missing.append("source_collection")
    if not destination_collection:
        missing.append("destination_collection")
    if not column:
        missing.append("column")
    if not subsection_column:
        missing.append("subsection_column")

    if missing:
        logger.warning("POST /create-statute-subsections - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    _warn_if_database_or_collection_not_found(
        database, source_collection, "POST /create-statute-subsections",
    )

    logger.info(
        "POST /create-statute-subsections - %s.%s -> %s.%s column=%s subsection_column=%s",
        database, source_collection, database, destination_collection,
        column, subsection_column,
    )

    try:
        db = mongo_client[database]
        source_coll = db[source_collection]
        dest_coll = db[destination_collection]
        docs = list(source_coll.find(source_query))
        del_filter = source_query if source_query else {}
        deleted = dest_coll.delete_many(del_filter)
        if deleted.deleted_count:
            logger.info(
                "POST /create-statute-subsections - Removed %d existing subsections matching source_query",
                deleted.deleted_count,
            )
    except Exception as e:
        logger.exception("POST /create-statute-subsections - Failed to read source collection")
        return jsonify({"error": f"Failed to read source collection: {e!s}"}), 500

    base_instructions = STATUTE_SUBSECTION_DEFAULT_PROMPT
    if parse_prompt.strip():
        base_instructions = base_instructions.rstrip() + "\n\nAdditional parsing instructions:\n" + parse_prompt.strip()

    records_inserted = 0
    source_rows_processed = 0
    source_rows_skipped = 0
    llm_errors = 0

    for doc in docs:
        raw = doc.get(column)
        text = str(raw) if raw is not None else ""
        if not text or not text.strip():
            source_rows_skipped += 1
            continue

        # Add line numbers for LLM to reference (1-indexed). Use same truncated text for extraction.
        statute_truncated = text[:50000]
        statute_lines = statute_truncated.split("\n")
        numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(statute_lines)]
        numbered_text = "\n".join(numbered_lines)

        user_message = f"""{base_instructions}

Statute section to parse (each line is numbered for reference):

<statute_section>
{numbered_text}
</statute_section>

Return only valid JSON with the subsections array. Use start_line and end_line for each subsection."""

        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                messages=[{"role": "user", "content": user_message}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            response_text = "".join(content)
        except Exception as e:
            logger.warning("POST /create-statute-subsections - LLM call failed for doc: %s", e)
            llm_errors += 1
            source_rows_skipped += 1
            continue

        raw_json = extract_json_block(response_text)
        if not raw_json:
            logger.warning("POST /create-statute-subsections - No JSON block in LLM response")
            llm_errors += 1
            source_rows_skipped += 1
            continue

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                import json5
                parsed = json5.loads(raw_json)
            except Exception as e:
                logger.warning("POST /create-statute-subsections - Invalid JSON from LLM: %s", e)
                llm_errors += 1
                source_rows_skipped += 1
                continue

        subsections = parsed.get("subsections")
        if not isinstance(subsections, list) or not subsections:
            source_rows_skipped += 1
            continue

        base = {k: v for k, v in doc.items() if k != column and k != "_id"}
        if "_id" in doc:
            base["source_id"] = str(doc["_id"])

        # Extract chunk header (first line starting with #) for prepending if LLM omits it
        chunk_header = ""
        first_line = text.split("\n")[0].strip() if text else ""
        if first_line.startswith("#"):
            chunk_header = first_line

        # Original lines (unnumbered) for text extraction - must match what we sent to LLM
        orig_lines = statute_lines
        n_lines = len(orig_lines)

        for sub in subsections:
            sub_id = sub.get("identifier", "")
            sub_text = sub.get("text", "")

            # Extract text from line range if start_line/end_line present
            start_line = sub.get("start_line")
            end_line = sub.get("end_line")
            if start_line is not None and end_line is not None:
                try:
                    start_idx = max(0, int(start_line) - 1)
                    end_idx = min(n_lines, int(end_line))
                    if start_idx < end_idx:
                        sub_text = " ".join(line.strip() for line in orig_lines[start_idx:end_idx] if line.strip())
                except (TypeError, ValueError):
                    pass

            if not sub_text and not sub_id:
                continue
            # Normalize linefeeds to spaces
            sub_text = " ".join(sub_text.split())
            # Prepend chunk header if present and subsection does not already start with it
            if chunk_header and sub_text and not sub_text.strip().startswith("#"):
                sub_text = f"{chunk_header} {sub_text}".strip()

            record = {
                **base,
                subsection_column: sub_text,
                "subsection_identifier": sub_id,
                "subchunk_id": str(uuid.uuid4()),
                column: text,
            }
            # Optional metadata from LLM
            if sub.get("header_text"):
                record["header_text"] = str(sub["header_text"]).strip()
            if sub.get("category"):
                record["category"] = str(sub["category"]).strip()
            if sub.get("category_reasoning"):
                record["category_reasoning"] = str(sub["category_reasoning"]).strip()
            try:
                dest_coll.insert_one(record)
                records_inserted += 1
            except Exception as e:
                logger.warning("POST /create-statute-subsections - Failed to insert: %s", e)
                break
        source_rows_processed += 1

    logger.info(
        "POST /create-statute-subsections - Inserted %d records, processed %d source rows, skipped %d, llm_errors=%d",
        records_inserted, source_rows_processed, source_rows_skipped, llm_errors,
    )
    return jsonify({
        "database": database,
        "source_collection": source_collection,
        "destination_collection": destination_collection,
        "column": column,
        "subsection_column": subsection_column,
        "records_inserted": records_inserted,
        "source_rows_processed": source_rows_processed,
        "source_rows_skipped": source_rows_skipped,
        "llm_errors": llm_errors,
    })


@core_bp.post("/create-statute-subtopics")
def create_statute_subtopics():
    """Identify compliance sub_topics from statute sections using LLM and write to destination collection.

    For each record in source_collection, reads the value from `column`, uses an LLM
    to identify distinct regulatory sub_topics (testable compliance requirements).
    Creates one record per sub_topic in destination_collection. Each destination
    record includes all source columns except the split column, plus subsection_column
    (the statute text for context), sub_topic, requirement_summary, policy_categories,
    requires_consent, consumer_facing, and subchunk_id.

    Optional parse_prompt: when provided, appended as additional parsing instructions.
    When blank, uses the default prompt for statute sub-topic extraction.
    """
    logger.info("POST /create-statute-subtopics - Starting")
    payload = request.get_json(silent=True) or {}
    database = payload.get("database")
    source_collection = payload.get("source_collection")
    destination_collection = payload.get("destination_collection")
    column = payload.get("column")
    subsection_column = payload.get("subsection_column")
    parse_prompt = payload.get("parse_prompt") or ""
    source_query_param = payload.get("source_query")
    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str) and source_query_param.strip():
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning("POST /create-statute-subtopics - Invalid JSON in source_query: %s", exc)
                return jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}), 400
        if not isinstance(source_query, dict):
            source_query = {}

    missing = []
    if not database:
        missing.append("database")
    if not source_collection:
        missing.append("source_collection")
    if not destination_collection:
        missing.append("destination_collection")
    if not column:
        missing.append("column")
    if not subsection_column:
        missing.append("subsection_column")

    if missing:
        logger.warning("POST /create-statute-subtopics - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    _warn_if_database_or_collection_not_found(
        database, source_collection, "POST /create-statute-subtopics",
    )

    logger.info(
        "POST /create-statute-subtopics - %s.%s -> %s.%s column=%s subsection_column=%s",
        database, source_collection, database, destination_collection,
        column, subsection_column,
    )

    try:
        db = mongo_client[database]
        source_coll = db[source_collection]
        dest_coll = db[destination_collection]
        docs = list(source_coll.find(source_query))
        del_filter = source_query if source_query else {}
        deleted = dest_coll.delete_many(del_filter)
        if deleted.deleted_count:
            logger.info(
                "POST /create-statute-subtopics - Removed %d existing subtopics matching source_query",
                deleted.deleted_count,
            )
    except Exception as e:
        logger.exception("POST /create-statute-subtopics - Failed to read source collection")
        return jsonify({"error": f"Failed to read source collection: {e!s}"}), 500

    base_instructions = STATUTE_SUBTOPIC_DEFAULT_PROMPT
    if parse_prompt.strip():
        base_instructions = base_instructions.rstrip() + "\n\nAdditional parsing instructions:\n" + parse_prompt.strip()

    records_inserted = 0
    source_rows_processed = 0
    source_rows_skipped = 0
    llm_errors = 0
    skip_empty_text = 0
    skip_empty_subtopics = 0

    logger.info("POST /create-statute-subtopics - Found %d source docs, column=%r", len(docs), column)

    # Common text column names for statute collections (for fallback when column is wrong)
    _text_col_candidates = ("sub_chunk_text", "chunk_text", "subchunk_text", "text")

    for doc in docs:
        raw = doc.get(column)
        if (raw is None or (isinstance(raw, str) and not raw.strip())) and doc:
            # Try fallback columns when requested column is empty
            for cand in _text_col_candidates:
                if cand != column:
                    alt = doc.get(cand)
                    if alt and (isinstance(alt, str) and alt.strip()):
                        logger.info(
                            "POST /create-statute-subtopics - Column %r empty, using fallback %r (doc has keys: %s)",
                            column, cand, list(doc.keys())[:12],
                        )
                        raw = alt
                        break
        text = str(raw) if raw is not None else ""
        if not text or not text.strip():
            skip_empty_text += 1
            source_rows_skipped += 1
            if skip_empty_text <= 3:
                logger.warning(
                    "POST /create-statute-subtopics - Skipping doc (empty %r): doc has keys %s",
                    column,
                    list(doc.keys())[:15],
                )
            continue

        category = doc.get("category")
        category_context = ""
        if category:
            category_context = f"\n\nThe statute section has been pre-classified into category: {category}."

        statute_truncated = text[:50000]
        statute_lines = statute_truncated.split("\n")
        numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(statute_lines)]
        numbered_text = "\n".join(numbered_lines)

        user_message = f"""{base_instructions}{category_context}

Statute section to analyze (each line is numbered for reference):

<statute_section>
{numbered_text}
</statute_section>

Return only valid JSON with the sub_topics array."""

        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                messages=[{"role": "user", "content": user_message}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            response_text = "".join(content)
        except Exception as e:
            logger.warning("POST /create-statute-subtopics - LLM call failed for doc: %s", e)
            llm_errors += 1
            source_rows_skipped += 1
            continue

        raw_json = extract_json_block(response_text)
        if not raw_json:
            logger.warning(
                "POST /create-statute-subtopics - No JSON block in LLM response (len=%d). First 200 chars: %r",
                len(response_text),
                (response_text[:200] + "..." if len(response_text) > 200 else response_text),
            )
            llm_errors += 1
            source_rows_skipped += 1
            continue

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                import json5
                parsed = json5.loads(raw_json)
            except Exception as e:
                logger.warning(
                    "POST /create-statute-subtopics - Invalid JSON from LLM: %s. Snippet: %r",
                    e,
                    raw_json[:300] if raw_json else None,
                )
                llm_errors += 1
                source_rows_skipped += 1
                continue

        sub_topics = parsed.get("sub_topics")
        if sub_topics is None:
            sub_topics = parsed.get("subtopics")
        if not isinstance(sub_topics, list) or not sub_topics:
            skip_empty_subtopics += 1
            source_rows_skipped += 1
            if skip_empty_subtopics <= 3:
                logger.warning(
                    "POST /create-statute-subtopics - Empty sub_topics for doc category=%r. Parsed keys: %s, sub_topics type=%s",
                    doc.get("category"),
                    list(parsed.keys()) if isinstance(parsed, dict) else "not-dict",
                    type(sub_topics).__name__ if sub_topics is not None else "None",
                )
            continue

        base = {k: v for k, v in doc.items() if k != column and k != "_id"}
        if "_id" in doc:
            base["source_id"] = str(doc["_id"])

        statute_text = " ".join(text.split())

        for st in sub_topics:
            sub_topic = st.get("sub_topic")
            if not sub_topic:
                continue
            record = {
                **base,
                subsection_column: statute_text,
                "sub_topic": str(sub_topic).strip(),
                "subchunk_id": str(uuid.uuid4()),
            }
            if doc.get("category") is not None:
                record["category"] = str(doc["category"]).strip()
            if st.get("requirement_summary") is not None:
                record["requirement_summary"] = str(st["requirement_summary"]).strip()
            if st.get("policy_categories") is not None:
                cats = st["policy_categories"]
                record["policy_categories"] = [str(c).strip() for c in cats] if isinstance(cats, list) else []
            if "requires_consent" in st:
                record["requires_consent"] = bool(st["requires_consent"])
            if "consumer_facing" in st:
                record["consumer_facing"] = bool(st["consumer_facing"])
            try:
                dest_coll.insert_one(record)
                records_inserted += 1
            except Exception as e:
                logger.warning("POST /create-statute-subtopics - Failed to insert: %s", e)
                break
        source_rows_processed += 1

    logger.info(
        "POST /create-statute-subtopics - Inserted %d records, processed %d source rows, skipped %d, llm_errors=%d (skip_empty_text=%d, skip_empty_subtopics=%d)",
        records_inserted, source_rows_processed, source_rows_skipped, llm_errors,
        skip_empty_text, skip_empty_subtopics,
    )
    return jsonify({
        "database": database,
        "source_collection": source_collection,
        "destination_collection": destination_collection,
        "column": column,
        "subsection_column": subsection_column,
        "records_inserted": records_inserted,
        "source_rows_processed": source_rows_processed,
        "source_rows_skipped": source_rows_skipped,
        "llm_errors": llm_errors,
    })


# Default prompt for policy subsection extraction when parse_prompt is blank.
# Splits policy sections into logical chunks, excluding headers and irrelevant text.
POLICY_SUBSECTION_DEFAULT_PROMPT = """
You are a privacy policy parser. Given the full text of a company's privacy
policy, segment it into distinct thematic sections and classify each one.

INSTRUCTIONS:

1. Read the entire policy text carefully before segmenting.

2. Identify natural section boundaries using these signals:
   - Explicit headings or subheadings in the text
   - Shifts in topic even when no heading is present
   - Numbered or lettered subsections that form a logical unit

3. When the policy has clear headings, respect them as boundaries. When it
   does not, infer boundaries based on topic shifts. Do not split a single
   coherent topic across multiple sections.

4. A single policy section may map to multiple categories. If so, assign the
   PRIMARY category based on the dominant topic, and list secondary categories
   in the "secondary_categories" field.

5. The document has line numbers at the start of each line (e.g., "1: ...", "2: ...").
   For each section, report start_line and end_line (1-indexed, inclusive) instead of
   copying the full text. Do not include a "text" field.

6. If a section does not fit any defined category, classify it as "other".

Respond with valid JSON matching this schema (use start_line and end_line; no "text" field):
{
  "company_name": "<inferred company name or 'Unknown'>",
  "sections": [
    {
      "section_index": 0,
      "heading": "<original heading if present, otherwise a short generated label>",
      "generated_heading": true,
      "category": "<primary category>",
      "secondary_categories": [],
      "start_line": 1,
      "end_line": 15
    }
  ]
}

CATEGORIES:
- "data_collection": What personal data is collected, sources of collection,
  categories of data gathered.

- "data_use": Purposes for processing personal data, legal bases for
  processing.

- "data_sharing": Third parties data is shared with, categories of recipients,
  sale of data, affiliate sharing.

- "consumer_rights": How consumers can exercise rights (access, deletion,
  correction, opt-out), verification procedures, response timelines.

- "data_retention": How long data is kept, retention criteria, deletion
  practices.

- "data_security": Security measures, safeguards, breach notification
  procedures.

- "children": COPPA compliance, age verification, parental consent.

- "cookies_tracking": Cookie usage, tracking technologies, advertising
  practices, opt-out mechanisms.

- "state_specific": State-by-state rights disclosures (California, Virginia,
  Kentucky, etc.)

- "contact": How to reach the company, DPO information, complaint procedures.

- "updates": Policy change notification practices, effective dates.
"""



POLICY_SUBSECTION_DEFAULT_PROMPT_OLD = """You are a privacy policy parser specializing in compliance-relevant content.

Task:
Parse the provided policy section into logical subsections (chunks). Each chunk should be a coherent paragraph or block of text that is relevant to privacy policies and compliance.

Instructions:
- Do NOT include text that is irrelevant to privacy policies and compliance.
- Split the body content into logical chunks by paragraph or semantic boundaries.
- Preserve the original language verbatim. Do NOT summarize or paraphrase.
- Maintain the original order of chunks.
- If the text has no clear chunk boundaries, return a single chunk with the relevant body text (excluding headers).

Output Format:
Return ONLY valid JSON with this exact structure (no surrounding text):
{
  "subsections": [
    {
      "identifier": "1",
      "text": "First logical chunk of body text..."
    },
    {
      "identifier": "2",
      "text": "Second logical chunk..."
    }
  ]
}"""


@core_bp.post("/create-policy-subsections")
def create_policy_subsections():
    """Split policy section column into subsections using LLM and write to destination collection.

    For each record in source_collection, reads the value from `column`, uses an LLM
    to identify policy subsections (logical chunks), excluding section headers and
    irrelevant text. Creates one record per subsection in destination_collection.
    Each destination record includes all source columns except the split column,
    plus subsection_column, subsection_identifier, and a unique subchunk_id (guid).

    Optional source_query: MongoDB query to filter source records.
    Optional parse_prompt: when provided, appended as additional parsing instructions.
    When blank, uses the default prompt for policy subsection extraction.
    """
    logger.info("POST /create-policy-subsections - Starting")
    payload, err = _get_json_payload_or_error()
    if err is not None:
        return err[0], err[1]
    payload = payload or {}
    database = payload.get("database")
    source_collection = payload.get("source_collection")
    destination_collection = payload.get("destination_collection")
    column = payload.get("column")
    subsection_column = payload.get("subsection_column")
    parse_prompt = payload.get("parse_prompt") or ""
    source_query_param = payload.get("source_query")
    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str) and source_query_param.strip():
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning("POST /create-policy-subsections - Invalid JSON in source_query: %s", exc)
                return jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}), 400
        if not isinstance(source_query, dict):
            source_query = {}

    missing = []
    if not database:
        missing.append("database")
    if not source_collection:
        missing.append("source_collection")
    if not destination_collection:
        missing.append("destination_collection")
    if not column:
        missing.append("column")
    if not subsection_column:
        missing.append("subsection_column")

    if missing:
        logger.warning("POST /create-policy-subsections - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    _warn_if_database_or_collection_not_found(
        database, source_collection, "POST /create-policy-subsections",
    )

    logger.info(
        "POST /create-policy-subsections - %s.%s -> %s.%s column=%s subsection_column=%s",
        database, source_collection, database, destination_collection,
        column, subsection_column,
    )

    try:
        db = mongo_client[database]
        source_coll = db[source_collection]
        dest_coll = db[destination_collection]
        docs = list(source_coll.find(source_query))
        del_filter = source_query if source_query else {}
        deleted = dest_coll.delete_many(del_filter)
        if deleted.deleted_count:
            logger.info(
                "POST /create-policy-subsections - Removed %d existing subsections matching source_query",
                deleted.deleted_count,
            )
    except Exception as e:
        logger.exception("POST /create-policy-subsections - Failed to read source collection")
        return jsonify({"error": f"Failed to read source collection: {e!s}"}), 500

    base_instructions = POLICY_SUBSECTION_DEFAULT_PROMPT
    if parse_prompt.strip():
        base_instructions = base_instructions.rstrip() + "\n\nAdditional parsing instructions:\n" + parse_prompt.strip()

    records_inserted = 0
    source_rows_processed = 0
    source_rows_skipped = 0
    llm_errors = 0

    for doc in docs:
        raw = doc.get(column)
        text = str(raw) if raw is not None else ""
        if not text or not text.strip():
            source_rows_skipped += 1
            continue

        # Add line numbers for LLM to reference (1-indexed). Use same truncated text for extraction.
        policy_truncated = text[:50000]
        policy_lines = policy_truncated.split("\n")
        numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(policy_lines)]
        numbered_text = "\n".join(numbered_lines)

        user_message = f"""{base_instructions}

Policy section to parse (each line is numbered for reference):

<policy_section>
{numbered_text}
</policy_section>

Return only valid JSON with the sections array. Use start_line and end_line for each section; do not include a "text" field."""

        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                messages=[{"role": "user", "content": user_message}],
            )
            content = []
            for block in response.content:
                if block.type == "text":
                    content.append(block.text)
            response_text = "".join(content)
        except Exception as e:
            logger.warning("POST /create-policy-subsections - LLM call failed for doc: %s", e)
            llm_errors += 1
            source_rows_skipped += 1
            continue

        raw_json = extract_json_block(response_text)
        if not raw_json:
            logger.warning("POST /create-policy-subsections - No JSON block in LLM response")
            llm_errors += 1
            source_rows_skipped += 1
            continue

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                import json5
                parsed = json5.loads(raw_json)
            except Exception as e:
                logger.warning("POST /create-policy-subsections - Invalid JSON from LLM: %s", e)
                llm_errors += 1
                source_rows_skipped += 1
                continue

        # Support both "sections" (new prompt) and "subsections" (legacy)
        subsections = parsed.get("sections") or parsed.get("subsections")
        if not isinstance(subsections, list) or not subsections:
            source_rows_skipped += 1
            continue

        base = {k: v for k, v in doc.items() if k != column and k != "_id"}
        if "_id" in doc:
            base["source_id"] = str(doc["_id"])

        orig_lines = policy_lines
        n_lines = len(orig_lines)

        for sub in subsections:
            sub_id = str(sub.get("identifier", "") or sub.get("section_index", "") or sub.get("heading", "")).strip()
            sub_text = sub.get("text", "")

            # Extract text from line range if start_line/end_line present
            start_line = sub.get("start_line")
            end_line = sub.get("end_line")
            if start_line is not None and end_line is not None:
                try:
                    start_idx = max(0, int(start_line) - 1)
                    end_idx = min(n_lines, int(end_line))
                    if start_idx < end_idx:
                        sub_text = " ".join(line.strip() for line in orig_lines[start_idx:end_idx] if line.strip())
                except (TypeError, ValueError):
                    pass

            if not sub_text and not sub_id:
                continue
            sub_text = " ".join(sub_text.split())
            record = {
                **base,
                subsection_column: sub_text,
                "subsection_identifier": sub_id,
                "subchunk_id": str(uuid.uuid4()),
                column: text,
            }
            if sub.get("heading"):
                record["heading"] = str(sub["heading"]).strip()
            # Category: prefer LLM's category when present, otherwise use source's category
            if sub.get("category"):
                record["category"] = str(sub["category"]).strip()
            elif doc.get("category") is not None:
                record["category"] = str(doc["category"]).strip()
            try:
                dest_coll.insert_one(record)
                records_inserted += 1
            except Exception as e:
                logger.warning("POST /create-policy-subsections - Failed to insert: %s", e)
                break
        source_rows_processed += 1

    logger.info(
        "POST /create-policy-subsections - Inserted %d records, processed %d source rows, skipped %d, llm_errors=%d",
        records_inserted, source_rows_processed, source_rows_skipped, llm_errors,
    )
    return jsonify({
        "database": database,
        "source_collection": source_collection,
        "destination_collection": destination_collection,
        "column": column,
        "subsection_column": subsection_column,
        "records_inserted": records_inserted,
        "source_rows_processed": source_rows_processed,
        "source_rows_skipped": source_rows_skipped,
        "llm_errors": llm_errors,
    })


# Index job service (for background sub-vector-index workflow)
_index_job_storage = None
_flask_app = None


def init_index_job(app, mongo):
    """Initialize index job service with Flask app and MongoDB client."""
    global _index_job_storage, _flask_app
    from compliance_config import load_config
    from index_job_service import IndexJobStorage

    config = load_config()
    _index_job_storage = IndexJobStorage(mongo, config)
    _flask_app = app


def _get_index_job_storage():
    if _index_job_storage is None:
        raise RuntimeError("Index job service not initialized; call init_index_job first")
    return _index_job_storage


@core_bp.post("/create-sub-vector-index")
def create_sub_vector_index():
    """Start a background job to create sub-vector indexes (subsections -> embeddings -> vector index).

    Accepts document_type ('policy' or 'statute') and source_query. Returns job_id immediately.
    Use GET /index-jobs/<job_id> to poll status.
    """
    logger.info("POST /create-sub-vector-index - Starting")
    raw = request.get_data(as_text=True) or ""
    payload = request.get_json(silent=True, force=True) or {}
    if not payload and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
            try:
                import json5
                payload = json5.loads(raw)
            except Exception:
                pass
            if not payload:
                # Fallback: fix single-quoted keys/values (common in Swagger/copied JSON)
                import re
                fixed = re.sub(r"'([^']*)'\s*:\s*'([^']*)'", r'"\1": "\2"', raw)
                fixed = re.sub(r"'\s*:\s*'", r'": "', fixed)
                try:
                    payload = json.loads(fixed)
                except json.JSONDecodeError:
                    payload = None
            if not payload:
                return jsonify({"error": "Invalid JSON in request body"}), 400
    document_type_raw = payload.get("document_type")
    source_query = payload.get("source_query") or {}

    if document_type_raw is None or not isinstance(document_type_raw, str):
        return jsonify({"error": "document_type is required and must be 'policy' or 'statute'"}), 400
    document_type = document_type_raw.strip().lower()
    if document_type not in ("policy", "statute"):
        return jsonify({"error": "document_type must be 'policy' or 'statute'"}), 400

    if isinstance(source_query, str) and source_query.strip():
        try:
            source_query = json.loads(source_query)
        except json.JSONDecodeError:
            return jsonify({"error": "source_query must be valid JSON when provided as string"}), 400
    if not isinstance(source_query, dict):
        source_query = {}

    try:
        from index_job_service import start_sub_vector_index_job

        job_storage = _get_index_job_storage()
        job_id = start_sub_vector_index_job(
            request_dict={"document_type": document_type, "source_query": source_query},
            job_storage=job_storage,
            flask_app=_flask_app,
        )
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@core_bp.get("/index-jobs/<job_id>")
def get_index_job(job_id: str):
    """Get index job status and result."""
    job_storage = _get_index_job_storage()
    job = job_storage.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@core_bp.post("/create-chunks")
def create_chunks():
    """Chunk source column with overlap and write to destination collection.

    For each record in source_collection, reads source_column, splits into
    overlapping chunks (chunk_size, overlap), and writes one record per chunk
    to destination_collection. Each record has all source columns except the
    source column, plus chunk_column (chunk text), source_id, and chunk_index.

    Example: database=privacy-compliance, source_collection=documents,
    destination_collection=chunks, chunk_size=1200, overlap=200,
    source_column=text, chunk_column=chunk_text.
    """
    logger.info("POST /create-chunks - Starting")
    payload = request.get_json(silent=True) or {}
    database = payload.get("database")
    source_collection = payload.get("source_collection")
    destination_collection = payload.get("destination_collection")
    chunk_size = payload.get("chunk_size", DEFAULT_CHUNK_SIZE)
    overlap = payload.get("overlap", DEFAULT_CHUNK_OVERLAP)
    source_column = payload.get("source_column")
    chunk_column = payload.get("chunk_column")

    missing = []
    if not database:
        missing.append("database")
    if not source_collection:
        missing.append("source_collection")
    if not destination_collection:
        missing.append("destination_collection")
    if not source_column:
        missing.append("source_column")
    if not chunk_column:
        missing.append("chunk_column")

    if missing:
        logger.warning("POST /create-chunks - Missing required parameters: %s", missing)
        return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400

    _warn_if_database_or_collection_not_found(
        database, source_collection, "POST /create-chunks",
    )

    try:
        chunk_size = int(chunk_size)
        overlap = int(overlap)
    except (TypeError, ValueError):
        return jsonify({"error": "chunk_size and overlap must be integers"}), 400

    logger.info(
        "POST /create-chunks - %s.%s -> %s.%s chunk_size=%d overlap=%d source_column=%s chunk_column=%s",
        database, source_collection, database, destination_collection,
        chunk_size, overlap, source_column, chunk_column,
    )

    try:
        db = mongo_client[database]
        source_coll = db[source_collection]
        dest_coll = db[destination_collection]
        docs = list(source_coll.find({}))
    except Exception as e:
        logger.exception("POST /create-chunks - Failed to read source collection")
        return jsonify({"error": f"Failed to read source collection: {e!s}"}), 500

    records_inserted = 0
    source_rows_processed = 0
    source_rows_skipped = 0

    for doc in docs:
        raw = doc.get(source_column)
        text = str(raw) if raw is not None else ""
        chunks_list = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks_list:
            source_rows_skipped += 1
            continue

        base = {k: v for k, v in doc.items() if k != source_column and k != "_id"}
        if "_id" in doc:
            base["source_id"] = str(doc["_id"])

        for idx, chunk_text_val in enumerate(chunks_list):
            record = {
                **base,
                chunk_column: chunk_text_val,
                "chunk_index": idx,
            }
            try:
                dest_coll.insert_one(record)
                records_inserted += 1
            except Exception as e:
                logger.warning("POST /create-chunks - Failed to insert: %s", e)
                break
        source_rows_processed += 1

    logger.info(
        "POST /create-chunks - Inserted %d records, processed %d source rows, skipped %d",
        records_inserted, source_rows_processed, source_rows_skipped,
    )
    return jsonify({
        "database": database,
        "source_collection": source_collection,
        "destination_collection": destination_collection,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source_column": source_column,
        "chunk_column": chunk_column,
        "records_inserted": records_inserted,
        "source_rows_processed": source_rows_processed,
        "source_rows_skipped": source_rows_skipped,
    })


@core_bp.post("/parse-llm")
def parse_llm():
    """Parse document text from a collection using LLM.

    Reads all records from the specified database/collection, concatenates
    all "text" attributes into one string, and uses this as input for the LLM
    along with the parse_prompt.
    """
    logger.info("POST /parse-llm - Starting LLM document parsing")
    payload = request.get_json(silent=True) or {}

    database = payload.get("database")
    collection = payload.get("collection")
    parse_prompt = payload.get("parse_prompt")
    document_id = payload.get("document_id")
    logger.info("POST /parse-llm - Parameters: database=%s, collection=%s, parse_prompt=%s, document_id=%s",
                database, collection, parse_prompt[:100] if parse_prompt else None, document_id)

    # Validate required parameters
    missing_params = []
    if not database:
        missing_params.append("database")
    if not collection:
        missing_params.append("collection")
    if not parse_prompt:
        missing_params.append("parse_prompt")

    if missing_params:
        logger.warning("POST /parse-llm - Missing required parameters: %s", ", ".join(missing_params))
        return jsonify({
            "error": f"Missing required parameters: {', '.join(missing_params)}"
        }), 400

    logger.info("POST /parse-llm - Reading documents from %s.%s", database, collection)

    # Read documents from the collection
    db = mongo_client[database]
    if document_id:
        # Query by document_id field (e.g. statutes) or _id (e.g. ingest).
        doc = db[collection].find_one({"document_id": document_id})
        if not doc:
            try:
                doc = db[collection].find_one({"_id": ObjectId(document_id)})
            except (InvalidId, TypeError):
                doc = db[collection].find_one({"_id": document_id})
        if not doc:
            logger.warning("POST /parse-llm - Document not found with document_id: %s", document_id)
            return jsonify({"error": f"Document not found with document_id: {document_id}"}), 400
        documents = [doc]
    else:
        # Query all documents in the collection
        documents = list(db[collection].find())

    if not documents:
        logger.warning("POST /parse-llm - No documents found in %s.%s", database, collection)
        return jsonify({"error": f"No documents found in {database}.{collection}"}), 400

    # Concatenate all "text" attributes
    text_parts = []
    for doc in documents:
        text = doc.get("text", "")
        if text and text.strip():
            text_parts.append(text.strip())

    if not text_parts:
        logger.warning("POST /parse-llm - No text content found in documents")
        return jsonify({"error": "No text content found in documents"}), 400

    combined_text = "\n\n".join(text_parts)
    logger.info("POST /parse-llm - Combined %d documents into %d characters",
                len(text_parts), len(combined_text))

    # Split into lines and create numbered version for LLM
    lines = combined_text.split("\n")
    numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
    numbered_text = "\n".join(numbered_lines)

    system_prompt = """You are a legal text parser specializing in statutory interpretation.

Task:
Parse the provided legal statute into a structured, machine-readable format by
identifying and extracting its hierarchical sections.

Instructions:
- Preserve the original statutory language verbatim.
- Do NOT summarize, paraphrase, or interpret the text.
- Do NOT infer missing structure; rely only on explicit markers in the text.
- Maintain the original order of sections.

Identify and extract the following elements when present:
- Jurisdiction (e.g., California, United States)
- Code name (e.g., Civil Code, Health and Safety Code)
- Section citation (e.g., § 1798.100)
- Title or Act name
- Chapter or Division
- Article
- Section number
- Subsection (e.g., (a), (b), (1), (A))
- Heading or caption
- Body text

Output Format:
Return a JSON array where each object represents the smallest logical statutory
unit (typically a section or subsection).

Each object must include:
- "jurisdiction": string
- "code_name": string
- "level": one of ["title", "chapter", "article", "section", "subsection"]
- "identifier": the official number or label (e.g., "§ 1798.100", "(a)",
  "Chapter 3")
- "heading": the heading text if present, otherwise null
- "text": the full statutory text for that unit
- "parent_identifier": the identifier of the immediately enclosing unit, or
  null if top-level

The document includes line numbers at the start of each line. Use those line
numbers to identify section boundaries. Call the identify_sections tool and
return an array of sections with:
- section
- code_name
- jurisdiction
- parsed_header_text
- start_line (1-indexed)."""

    user_message = f"""Analyze the following document and identify the section boundaries according to the parsing instructions.

<document>
{numbered_text}
</document>

Exclude irrelevant sections from the results.
Any sections that do not pertain to company privacy policy should be omitted.

<parse_prompt>
{parse_prompt}
</parse_prompt>

Use the identify_sections tool to report the sections you identified."""

    # Define the tool for section identification
    tools = [
        {
            "name": "identify_sections",
            "description": "Report the identified sections in the document with their starting line numbers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "description": "Array of identified sections",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {
                                    "type": "string",
                                    "description": "The formal section citation (e.g., \u00a7 1798.100)"
                                },
                                "code_name": {
                                    "type": "string",
                                    "description": "The legal code name (e.g., Civil Code)"
                                },
                                "jurisdiction": {
                                    "type": "string",
                                    "description": "The jurisdiction for this section (e.g., California)"
                                },
                                "parsed_header_text": {
                                    "type": "string",
                                    "description": "The header, title, or identifying text for this section"
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": "The 1-indexed line number where this section begins"
                                }
                            },
                            "required": [
                                "section",
                                "code_name",
                                "jurisdiction",
                                "parsed_header_text",
                                "start_line"
                            ]
                        }
                    }
                },
                "required": ["sections"]
            }
        }
    ]

    try:
        logger.info("POST /parse-llm - Calling Anthropic API with tool use (model: claude-sonnet-4-6)")
        with anthropic_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=system_prompt,
            tools=tools,
            tool_choice={"type": "tool", "name": "identify_sections"}
        ) as stream:
            response = stream.get_final_message()

        # Extract tool use from response
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "identify_sections":
                tool_use_block = block
                break

        if not tool_use_block:
            logger.error("POST /parse-llm - LLM did not use the identify_sections tool")
            return jsonify({"error": "LLM did not return section identification"}), 500

        sections = tool_use_block.input.get("sections", [])
        logger.info("POST /parse-llm - LLM identified %d sections", len(sections))

        # Sort sections by start_line
        sections.sort(key=lambda s: s.get("start_line", 0))

        # Build parsed_doc by extracting text between line numbers
        parsed_doc = []
        total_lines = len(lines)

        for i, section in enumerate(sections):
            start_line = section.get("start_line", 1)
            # Determine end line (next section's start - 1, or end of document)
            if i + 1 < len(sections):
                end_line = sections[i + 1].get("start_line", total_lines + 1) - 1
            else:
                end_line = total_lines

            # Convert to 0-indexed and extract lines
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, end_line)
            section_lines = lines[start_idx:end_idx]
            parsed_text = " ".join(" ".join(section_lines).split()).strip()

            parsed_doc.append({
                "section": section.get("section"),
                "code_name": section.get("code_name"),
                "jurisdiction": section.get("jurisdiction"),
                "parsed_header_text": " ".join((section.get("parsed_header_text") or "").split()),
                "parsed_text": parsed_text
            })

        logger.info("POST /parse-llm - Successfully parsed document into %d sections", len(parsed_doc))
        return jsonify({"parsed_doc": parsed_doc})

    except Exception as e:
        logger.error("POST /parse-llm - LLM parsing failed: %s", e)
        return jsonify({"error": f"LLM parsing failed: {str(e)}"}), 500


def _fetch_document_text(database_name: str, collection_name: str, document_id: str):
    """Fetch document from MongoDB and extract text.

    Looks up by document_id field first, then _id (ObjectId or string).
    Extracts text from: doc.text; else doc.chunk_text or doc.section_text;
    else concatenate doc.policy_chunks (chunk_text or chunk_header_text).

    Returns:
        tuple: (text, None) on success, or (None, (response, status_code)) on error.
    """
    db = mongo_client[database_name]
    coll = db[collection_name]
    doc = coll.find_one({"document_id": document_id})
    if not doc:
        try:
            doc = coll.find_one({"_id": ObjectId(document_id)})
        except (InvalidId, TypeError):
            doc = coll.find_one({"_id": document_id})
    if not doc:
        return None, (jsonify({"error": "Document not found"}), 404)

    text = (doc.get("text") or "").strip()
    if text:
        return text, None

    chunk_text = (doc.get("chunk_text") or doc.get("section_text") or "").strip()
    if chunk_text:
        return chunk_text, None

    chunks = doc.get("policy_chunks")
    if isinstance(chunks, list):
        parts = []
        for c in chunks:
            if isinstance(c, dict):
                part = (c.get("chunk_text") or c.get("chunk_header_text") or "").strip()
                if part:
                    parts.append(part)
        if parts:
            return "\n\n".join(parts), None

    return None, (jsonify({"error": "Document has no extractable text"}), 404)


GAP_CHECK_PROMPT_TEMPLATE = """Consider the following statute requirement:

{statute_text}

Below is a privacy policy document. Does the policy address this statute requirement? If yes, quote the exact policy phrase that addresses it in policy_quote. If there is a conflict with the statute, describe it in conflict_description and, if the policy contains a phrase that conflicts, quote that exact phrase in policy_quote. policy_quote must be a verbatim substring of the policy text.

PRIVACY POLICY:
{policy_text}

Respond with only a valid JSON object. Use these exact keys: addressed, policy_quote, missing, conflict, conflict_description. addressed, missing, and conflict must be booleans. policy_quote and conflict_description must be strings or null if not applicable."""


@core_bp.post("/gap-check")
def gap_check():
    """Single statute-to-policy gap check via LLM.

    Fetches both statute and policy from MongoDB, invokes the LLM, and returns
    a structured gap_check result (addressed, policy_quote, missing, conflict,
    conflict_description).
    """
    logger.info("POST /gap-check - Starting gap check")
    payload = request.get_json(silent=True) or {}

    policy_database_name = payload.get("policy_database_name")
    policy_collection_name = payload.get("policy_collection_name")
    policy_document_id = payload.get("policy_document_id")
    statute_database_name = payload.get("statute_database_name")
    statute_collection_name = payload.get("statute_collection_name")
    statute_document_id = payload.get("statute_document_id")
    max_policy_chars = int(payload.get("max_policy_chars", 8000))

    logger.info(
        "POST /gap-check - policy: %s.%s id=%s, statute: %s.%s id=%s",
        policy_database_name,
        policy_collection_name,
        policy_document_id,
        statute_database_name,
        statute_collection_name,
        statute_document_id,
    )

    # Validate policy params
    if not policy_database_name or not policy_collection_name or not policy_document_id:
        return jsonify({
            "error": "policy_database_name, policy_collection_name, and policy_document_id are required"
        }), 400

    # Validate statute params
    if not statute_database_name or not statute_collection_name or not statute_document_id:
        return jsonify({
            "error": "statute_database_name, statute_collection_name, and statute_document_id are required"
        }), 400

    # Fetch statute text
    statute_text, err = _fetch_document_text(
        statute_database_name, statute_collection_name, statute_document_id
    )
    if err is not None:
        resp, code = err
        if code == 404:
            return jsonify({"error": "Statute document not found"}), 404
        return resp, code

    # Fetch policy text
    policy_text, err = _fetch_document_text(
        policy_database_name, policy_collection_name, policy_document_id
    )
    if err is not None:
        resp, code = err
        if code == 404:
            return jsonify({"error": "Policy document not found"}), 404
        return resp, code

    # Truncate statute to 4000 chars (keep start)
    if len(statute_text) > 4000:
        statute_text = statute_text[:4000]

    # Truncate policy: prefer keeping end (disclosure sections). Use last N chars.
    if len(policy_text) > max_policy_chars:
        policy_text = policy_text[-max_policy_chars:]

    prompt = GAP_CHECK_PROMPT_TEMPLATE.format(
        statute_text=statute_text,
        policy_text=policy_text,
    )

    def _parse_gap_response(raw_text: str):
        raw = extract_json_block(raw_text)
        if not raw:
            return None
        try:
            out = json.loads(raw)
            return {
                "addressed": bool(out.get("addressed")),
                "policy_quote": out.get("policy_quote") if out.get("policy_quote") else None,
                "missing": bool(out.get("missing", True)),
                "conflict": bool(out.get("conflict")),
                "conflict_description": out.get("conflict_description") if out.get("conflict_description") else None,
            }
        except (json.JSONDecodeError, TypeError):
            return None

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        content = []
        for block in response.content:
            if block.type == "text":
                content.append(block.text)
        raw_text = "".join(content)

        result = _parse_gap_response(raw_text)
        if result is None:
            # Retry with stricter prompt
            retry_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY a valid JSON object, no other text. Use the exact keys: addressed, policy_quote, missing, conflict, conflict_description."
            retry_response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": retry_prompt}],
            )
            retry_content = []
            for block in retry_response.content:
                if block.type == "text":
                    retry_content.append(block.text)
            result = _parse_gap_response("".join(retry_content))

        if result is None:
            logger.error("POST /gap-check - Could not parse LLM response")
            return jsonify({"error": "Could not parse SLM response"}), 500

        logger.info("POST /gap-check - Success")
        return jsonify({"gap_check": result})

    except Exception as e:
        logger.error("POST /gap-check - LLM call failed: %s", e)
        return jsonify({"error": f"LLM parsing failed: {str(e)}"}), 500


@core_bp.post("/crawl")
def crawl():
    """Crawl a URL with specified depth and breadth, returning combined page text.

    Parameters:
        url (str): The URL to start crawling from (required)
        depth (int): How deep to follow links from the starting URL (default: 2)
        breadth (int): Maximum number of pages to crawl (default: 10)

    Returns:
        JSON with combined text from all crawled pages, page count, and URLs visited.
    """
    logger.info("POST /crawl - Starting web crawl")
    payload = request.get_json(silent=True) or {}

    url = payload.get("url")
    depth = payload.get("depth", 2)
    breadth = payload.get("breadth", 10)
    logger.info("POST /crawl - Parameters: url=%s, depth=%s, breadth=%s", url, depth, breadth)

    # Validate required parameter
    if not url:
        logger.warning("POST /crawl - Missing required parameter: url")
        return jsonify({"error": "url is required"}), 400

    # Validate and convert depth and breadth to integers
    try:
        depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be at least 1")
    except (TypeError, ValueError) as e:
        logger.warning("POST /crawl - Invalid depth parameter: %s", depth)
        return jsonify({"error": f"depth must be a positive integer: {e}"}), 400

    try:
        breadth = int(breadth)
        if breadth < 1:
            raise ValueError("breadth must be at least 1")
    except (TypeError, ValueError) as e:
        logger.warning("POST /crawl - Invalid breadth parameter: %s", breadth)
        return jsonify({"error": f"breadth must be a positive integer: {e}"}), 400

    logger.info("POST /crawl - Crawling URL: %s (depth=%d, breadth=%d)", url, depth, breadth)

    pages = []
    crawl_method = None
    browser_errors = []

    # Try Playwright, then Puppeteer, then Selenium
    for method_name, crawl_fn in [
        ("playwright", playwright_crawl),
        ("puppeteer", puppeteer_crawl),
        ("selenium", selenium_crawl),
    ]:
        logger.info("POST /crawl - Attempting %s crawl", method_name)
        try:
            candidate_pages = crawl_fn(url, depth, breadth)
            if candidate_pages:
                pages = candidate_pages
                crawl_method = method_name
                logger.info("POST /crawl - %s crawl succeeded with %d pages", method_name, len(pages))
                break
            logger.warning("POST /crawl - %s returned no pages", method_name)
            browser_errors.append(f"{method_name}: no pages returned")
        except Exception as exc:
            logger.warning("POST /crawl - %s crawl failed: %s", method_name, exc)
            browser_errors.append(f"{method_name}: {exc}")

    # Fall back to Firecrawl if all browser methods failed or returned no pages
    if not pages:
        logger.info("POST /crawl - Attempting Firecrawl as fallback")
        try:
            crawl_result = firecrawl_client.crawl(
                url=url,
                limit=breadth,
                max_discovery_depth=depth,
            )

            # Debug: log crawl result structure
            logger.info("POST /crawl - crawl_result type: %s", type(crawl_result))
            if hasattr(crawl_result, '__dict__'):
                logger.info("POST /crawl - crawl_result attrs: %s", list(vars(crawl_result).keys()))
            if hasattr(crawl_result, 'data'):
                logger.info("POST /crawl - crawl_result.data: type=%s, len=%s",
                           type(crawl_result.data), len(crawl_result.data) if crawl_result.data else 0)
            if hasattr(crawl_result, 'status'):
                logger.info("POST /crawl - crawl_result.status: %s", crawl_result.status)

            pages = normalize_results(crawl_result)
            if pages:
                crawl_method = "firecrawl"
                logger.info("POST /crawl - Firecrawl succeeded with %d pages", len(pages))
        except Exception as exc:
            logger.error("POST /crawl - Firecrawl also failed for URL %s: %s", url, exc)
            error_msg = f"crawl failed - {'; '.join(browser_errors)}; firecrawl: {exc}"
            return jsonify({"error": error_msg}), 500

    page_count = len(pages)
    logger.info("POST /crawl - Crawl returned %d pages using %s", page_count, crawl_method)

    if page_count == 0:
        logger.warning("POST /crawl - Crawl returned 0 pages for URL: %s", url)
        return jsonify({"error": "crawl returned no pages - website may be blocking crawlers"}), 400

    # Extract URLs from crawled pages
    urls_crawled = []
    for page in pages:
        if hasattr(page, "metadata") and page.metadata:
            page_url = getattr(page.metadata, "url", None)
        elif isinstance(page, dict):
            page_url = page.get("url")
        else:
            page_url = get_page_attr(page, "url")
        if page_url:
            urls_crawled.append(page_url)

    # Combine pages - handle browser dicts (playwright/puppeteer/selenium) vs Firecrawl objects
    if crawl_method in ("playwright", "puppeteer", "selenium"):
        # Puppeteer returns dicts, combine directly
        combined_parts = []
        for p in pages:
            title = p.get("title", "")
            page_url = p.get("url", "")
            content = p.get("markdown", "")
            if content:
                header = "Source"
                if title:
                    header = f"{title} | Source"
                combined_parts.append(f"{header}: {page_url}\n{content}".strip())
        combined_text = "\n\n".join(combined_parts).strip()
    else:
        combined_text = combine_pages(pages)

    if not combined_text:
        logger.warning("POST /crawl - Crawl returned no content for URL: %s", url)
        return jsonify({"error": "crawl returned no content"}), 400

    logger.info("POST /crawl - Combined text length: %d characters from %d pages (method=%s)",
                len(combined_text), page_count, crawl_method)

    return jsonify({
        "url": url,
        "depth": depth,
        "breadth": breadth,
        "pages_crawled": page_count,
        "urls_crawled": urls_crawled,
        "combined_text": combined_text,
        "text_length": len(combined_text),
        "crawl_method": crawl_method,
    })
