"""Core endpoints for the Web Gather API.

Includes endpoints: gather, ingest, parse-llm, search, index, and crawl.
"""
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
from db import (
    WEB_GATHER_DB,
    DOCUMENTS_COLLECTION,
    get_embedding_model_name,
    utc_now,
)

logger = logging.getLogger("web-gather-api")

# Will be initialized by app.py
mongo_client = None
firecrawl_client = None
anthropic_client = None


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
                content = await page.evaluate('''() => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, noscript');
                    scripts.forEach(el => el.remove());

                    // Get text content
                    const body = document.body;
                    if (!body) return '';

                    // Get text with some structure preserved
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
                }''')

                if content:
                    pages.append({
                        'url': current_url,
                        'title': title,
                        'markdown': content
                    })
                    logger.info("Puppeteer: Extracted %d chars from %s", len(content), current_url)

                # If we haven't reached max depth, extract links to follow
                if current_depth < depth - 1 and len(pages) < breadth:
                    links = await page.evaluate('''() => {
                        const anchors = document.querySelectorAll('a[href]');
                        return Array.from(anchors)
                            .map(a => a.href)
                            .filter(href => href && href.startsWith('http'));
                    }''')

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

    # Look up embedding model using the index database name
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
            "Document loaded successfully. Use /vector-index endpoint to create "
            "vector embeddings."
        ),
    }

    if mode == "overwrite":
        response_data["overwritten"] = overwritten
        response_data["previous_document_count"] = previous_document_count

    logger.info("POST /ingest - Document loaded successfully (mode: %s): %s", mode, document_id)
    return jsonify(response_data)


@core_bp.post("/vector-index")
def index_document():
    """Index collection rows using the chunk_text field."""
    logger.info("POST /vector-index - Starting document indexing")
    payload = request.get_json(silent=True) or {}

    # Get required parameters
    source_database_name = payload.get("source_database_name")
    source_collection_name = payload.get("source_collection_name")
    index_database_name = payload.get("index_database_name")
    index_collection_name = payload.get("index_collection_name")
    source_query_param = payload.get("source_query")

    source_query = {}
    if source_query_param is not None:
        if isinstance(source_query_param, dict):
            source_query = source_query_param
        elif isinstance(source_query_param, str):
            try:
                source_query = json.loads(source_query_param)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "POST /vector-index - Invalid JSON in source_query: %s",
                    str(exc),
                )
                return (
                    jsonify({"error": f"Invalid JSON in source_query: {str(exc)}"}),
                    400,
                )
        else:
            logger.warning("POST /vector-index - source_query must be a JSON object")
            return jsonify({"error": "source_query must be a JSON object"}), 400

        if not isinstance(source_query, dict):
            logger.warning("POST /vector-index - source_query must be a JSON object")
            return jsonify({"error": "source_query must be a JSON object"}), 400

    logger.info(
        "POST /vector-index - Parameters: source_database_name=%s, "
        "source_collection_name=%s, index_database_name=%s, "
        "index_collection_name=%s",
        source_database_name,
        source_collection_name,
        index_database_name,
        index_collection_name,
    )
    if source_query:
        logger.info(
            "POST /vector-index - source_query for %s.%s: %s",
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
            "POST /vector-index - Missing required parameters: %s",
            ", ".join(missing_params),
        )
        return jsonify({
            "error": f"Missing required parameters: {', '.join(missing_params)}"
        }), 400

    logger.info(
        "POST /vector-index - Source: %s.%s",
        source_database_name,
        source_collection_name,
    )
    logger.info(
        "POST /vector-index - Index target: %s.%s",
        index_database_name,
        index_collection_name,
    )

    # Check if embedding model is configured for index_database_name
    model_name = get_embedding_model_name(index_database_name)
    if not model_name:
        logger.error(
            "POST /vector-index - No embedding model configured for database: %s",
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
            "POST /vector-index - No documents found in %s.%s",
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
        raw_chunk_text = doc.get("chunk_text")
        chunk_text = str(raw_chunk_text) if raw_chunk_text is not None else ""
        if not chunk_text.strip():
            skipped_rows += 1
            continue
        docs_to_index.append(doc)
        chunk_texts.append(chunk_text)

    if not docs_to_index:
        logger.warning(
            "POST /vector-index - No rows with chunk_text found in %s.%s",
            source_database_name,
            source_collection_name,
        )
        return jsonify({"error": "no rows with chunk_text to index"}), 400

    logger.info("POST /vector-index - Loading embedding model: %s", model_name)
    model = get_model(model_name)

    logger.info(
        "POST /vector-index - Generating embeddings for %d rows",
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
            index_collection.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "embedding": embedding.tolist(),
                        "chunk_text": chunk_text,
                        "indexed_at": utc_now(),
                        "source_id": str(doc.get("_id")),
                        "source_database_name": source_database_name,
                        "source_collection_name": source_collection_name,
                    }
                },
            )
    else:
        logger.info(
            "POST /vector-index - Clearing existing index rows for %s.%s in %s.%s",
            source_database_name,
            source_collection_name,
            index_database_name,
            index_collection_name,
        )
        index_collection.delete_many({
            "source_database_name": source_database_name,
            "source_collection_name": source_collection_name,
        })
        index_docs = []
        for doc, embedding, chunk_text in zip(docs_to_index, embeddings, chunk_texts):
            base_doc = {key: value for key, value in doc.items() if key != "_id"}
            index_docs.append(
                {
                    **base_doc,
                    "source_id": str(doc.get("_id")),
                    "chunk_text": chunk_text,
                    "embedding": embedding.tolist(),
                    "indexed_at": utc_now(),
                    "source_database_name": source_database_name,
                    "source_collection_name": source_collection_name,
                }
            )
        logger.info(
            "POST /vector-index - Inserting %d rows into %s.%s",
            len(index_docs),
            index_database_name,
            index_collection_name,
        )
        index_collection.insert_many(index_docs)

    logger.info(
        "POST /vector-index - Successfully indexed %d rows",
        len(docs_to_index),
    )
    return jsonify(
        {
            "source_database_name": source_database_name,
            "source_collection_name": source_collection_name,
            "index_database_name": index_database_name,
            "index_collection_name": index_collection_name,
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
        # Query for specific document by _id
        try:
            doc = db[collection].find_one({"_id": ObjectId(document_id)})
        except Exception as e:
            logger.warning("POST /parse-llm - Invalid document_id format: %s", document_id)
            return jsonify({"error": f"Invalid document_id format: {document_id}"}), 400
        if not doc:
            logger.warning("POST /parse-llm - Document not found with _id: %s", document_id)
            return jsonify({"error": f"Document not found with _id: {document_id}"}), 400
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
        logger.info("POST /parse-llm - Calling Anthropic API with tool use (model: claude-3-5-haiku-20241022)")
        with anthropic_client.messages.stream(
            model="claude-3-5-haiku-20241022",
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
            parsed_text = "\n".join(section_lines).strip()

            parsed_doc.append({
                "section": section.get("section"),
                "code_name": section.get("code_name"),
                "jurisdiction": section.get("jurisdiction"),
                "parsed_header_text": section.get("parsed_header_text", ""),
                "parsed_text": parsed_text
            })

        logger.info("POST /parse-llm - Successfully parsed document into %d sections", len(parsed_doc))
        return jsonify({"parsed_doc": parsed_doc})

    except Exception as e:
        logger.error("POST /parse-llm - LLM parsing failed: %s", e)
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
    puppeteer_error = None

    # Try Puppeteer first
    logger.info("POST /crawl - Attempting Puppeteer crawl first")
    try:
        puppeteer_pages = puppeteer_crawl(url, depth, breadth)
        if puppeteer_pages:
            pages = puppeteer_pages
            crawl_method = "puppeteer"
            logger.info("POST /crawl - Puppeteer crawl succeeded with %d pages", len(pages))
        else:
            logger.warning("POST /crawl - Puppeteer returned no pages, falling back to Firecrawl")
            puppeteer_error = "no pages returned"
    except Exception as exc:
        logger.warning("POST /crawl - Puppeteer crawl failed: %s, falling back to Firecrawl", exc)
        puppeteer_error = str(exc)

    # Fall back to Firecrawl if Puppeteer failed or returned no pages
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
            error_msg = f"crawl failed - puppeteer: {puppeteer_error}, firecrawl: {exc}"
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

    # Combine pages - handle both Puppeteer dicts and Firecrawl objects
    if crawl_method == "puppeteer":
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
