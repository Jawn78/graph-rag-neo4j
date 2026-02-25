"""
Document ingestion pipeline.

Optimizations:
- ProcessPoolExecutor for CPU-bound file parsing (avoids GIL bottleneck)
- Asyncio for I/O-bound embedding requests with backpressure control
- Streaming ingestion to limit memory usage
- Batch cache lookups to reduce SQLite queries
- Shared pre-compiled regex patterns from utils/text.py
"""

import os
import re
import glob
import hashlib
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Iterator, Optional

from neo4j import Driver

from ..config import embed_client, EMBED_MODEL
from ..graph.upsert import upsert_docs, upsert_chunks
from ..utils.text import sanitize_text, is_corrupted, split_sentences
from ..utils.logging import get_trace_id, set_trace_id, timed

logger = logging.getLogger(__name__)

# ---------- Optional readers ----------
PdfReader: Any = None
_HAS_PYPDF = False
try:
    from pypdf import PdfReader as _PdfReader
    PdfReader = _PdfReader
    _HAS_PYPDF = True
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader
        PdfReader = _PdfReader
        _HAS_PYPDF = True
    except Exception:
        pass

docx2txt: Any = None
_HAS_DOCX2TXT = False
try:
    import docx2txt as _docx2txt
    docx2txt = _docx2txt
    _HAS_DOCX2TXT = True
except Exception:
    pass

_HAS_CHARDET = False
try:
    import chardet
    _HAS_CHARDET = True
except Exception:
    pass

# ---------- Config ----------
# Allowed file extensions for ingestion
_ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".md", ".html", ".htm", ".rtf", ".csv"}

# Worker configuration
# Use ProcessPoolExecutor for CPU-bound parsing (avoids GIL)
PARSE_WORKERS = int(os.getenv("PARSE_WORKERS", "8"))
# Use asyncio with semaphore for I/O-bound embedding
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Chunking configuration
# Target ~1800 chars per chunk to fit comfortably in context windows
# Overlap of 200 chars maintains context across chunk boundaries
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Maximum text length to send to embedding model
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "12000"))

# Streaming batch size: process this many documents at a time to limit memory
STREAM_BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "50"))

# Heading detection patterns (pre-compiled)
_HEADING_PATTERNS = [
    re.compile(r"^\s*\d+(\.\d+)+\s+.{2,120}$"),      # 11.5. Dental Care
    re.compile(r"^\s*#{1,3}\s+.{2,120}$"),           # Markdown headings
    re.compile(r"^\s*[A-Z][A-Z0-9 /&\-]{3,100}\s*$") # ALL CAPS line
]


# ---------- Helpers ----------
def _sha1(s: str) -> str:
    """Generate SHA1 hash of string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks at sentence boundaries.

    Args:
        text: The text to chunk
        target_chars: Target size for each chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    sents = split_sentences(text)
    chunks, cur = [], ""

    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target_chars:
            cur += " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    # Add overlap from previous chunk
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            out.append((tail + " " + c).strip())

    return out


def _read_text_naive(path: str) -> str:
    """Read text file with encoding detection."""
    if _HAS_CHARDET:
        with open(path, "rb") as f:
            raw = f.read()
        import chardet
        enc = (chardet.detect(raw).get("encoding") or "utf-8").strip() or "utf-8"
        try:
            text = raw.decode(enc, errors="ignore")
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return sanitize_text(text)


def _extract_text_impl(path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from a file. Called in separate process.

    Returns (text, metadata) tuple.
    """
    ext = os.path.splitext(path)[1].lower()
    meta: Dict[str, Any] = {
        "source": "file",
        "path": path,
        "filename": os.path.basename(path),
        "ext": ext,
    }

    try:
        if ext == ".pdf" and _HAS_PYPDF:
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                try:
                    page_text = p.extract_text() or ""
                    pages.append(page_text)
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
            return sanitize_text(text), meta

        if ext == ".docx" and _HAS_DOCX2TXT:
            text = docx2txt.process(path) or ""
            return sanitize_text(text), meta

        if ext in (".txt", ".md", ".csv", ".rtf", ".html", ".htm"):
            return _read_text_naive(path), meta

        # Fallback for other extensions
        return _read_text_naive(path), meta

    except Exception as e:
        logger.warning(f"Failed to extract text from {path}: {e}")
        return "", meta


def _collect_files(folder: str) -> List[str]:
    """Collect all files with allowed extensions from folder."""
    if not os.path.isdir(folder):
        return []
    files: List[str] = []
    for ext in sorted(_ALLOWED_EXTS):
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return files


def _looks_like_heading(line: str) -> bool:
    """Check if a line looks like a heading."""
    ln = line.strip()
    if not ln or len(ln) > 140:
        return False

    for rx in _HEADING_PATTERNS:
        if rx.match(ln):
            return True

    # Check for uppercase-ish lines
    letters = [c for c in ln if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
        return True

    return False


def _first_heading_in_block(block: str) -> str:
    """Extract first heading from a text block."""
    for line in block.splitlines():
        if _looks_like_heading(line):
            return re.sub(r"\s+", " ", line.strip())
    return ""


# ---------- Parsing with ProcessPoolExecutor (CPU-bound) ----------
def _parse_files_parallel(paths: List[str], workers: int = PARSE_WORKERS) -> List[Dict[str, Any]]:
    """
    Parse files in parallel using ProcessPoolExecutor.

    Uses separate processes to avoid GIL bottleneck for CPU-intensive
    PDF/DOCX parsing.
    """
    docs: List[Dict[str, Any]] = []
    if not paths:
        return docs

    trace_id = get_trace_id()
    logger.info(f"[{trace_id}] Parsing {len(paths)} files with {workers} workers")

    # Use ProcessPoolExecutor for CPU-bound work to avoid GIL
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_text_impl, p): p for p in paths}

        for fut in as_completed(futures):
            path = futures[fut]
            try:
                text, meta = fut.result()
                if not text or not text.strip():
                    continue
                if is_corrupted(text):
                    logger.warning(f"[{trace_id}] Skipping corrupted file: {path}")
                    continue

                title = os.path.splitext(os.path.basename(meta["filename"]))[0]
                did = _sha1(title + ":" + text[:2000])
                docs.append({
                    "doc_id": did,
                    "title": title,
                    "text": text,
                    "metadata": meta,
                })
            except Exception as e:
                logger.warning(f"[{trace_id}] Error parsing {path}: {e}")

    logger.info(f"[{trace_id}] Parsed {len(docs)} documents successfully")
    return docs


def _build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build chunks from documents with heading extraction.

    Text is already sanitized from parsing stage - no re-sanitization needed.
    """
    out: List[Dict[str, Any]] = []
    current_heading = ""

    for d in docs:
        parts = _chunk_text(d["text"])
        for i, ct in enumerate(parts):
            # Truncate if needed (text already sanitized)
            ct = ct[:EMBED_MAX_CHARS]
            if not ct:
                continue

            # Extract or inherit heading
            h = _first_heading_in_block(ct) or current_heading
            if h:
                current_heading = h

            cid = _sha1(f"{d['doc_id']}:{i}:{ct[:100]}")
            out.append({
                "chunk_id": cid,
                "doc_id": d["doc_id"],
                "order": i,
                "text": ct,
                "heading": current_heading,
                "embedding": None,
                "metadata": {"from_doc": d["doc_id"]},
            })

    return out


# ---------- Embedding with asyncio + backpressure ----------
async def _embed_batch_async(chunks: List[Dict[str, Any]], semaphore: asyncio.Semaphore) -> Dict[str, List[float]]:
    """
    Embed a batch of chunks asynchronously.

    Returns dict of chunk_id -> embedding.
    """
    async with semaphore:
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        try:
            # Run synchronous embedding call in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: embed_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=texts,
                    encoding_format="float",
                )
            )

            result = {}
            for i, embedding_data in enumerate(response.data):
                result[chunk_ids[i]] = embedding_data.embedding
            return result

        except Exception as e:
            logger.warning(f"Batch embedding failed, trying individual: {e}")
            # Fallback to individual requests
            result = {}
            for chunk in chunks:
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda c=chunk: embed_client.embeddings.create(
                            model=EMBED_MODEL,
                            input=c["text"],
                            encoding_format="float",
                        )
                    )
                    result[chunk["chunk_id"]] = response.data[0].embedding
                except Exception as e2:
                    logger.warning(f"Failed to embed chunk {chunk['chunk_id'][:8]}: {e2}")
            return result


async def _embed_chunks_async(chunks: List[Dict[str, Any]]) -> None:
    """
    Embed all chunks asynchronously with backpressure control.

    Uses:
    - Batch cache lookup to minimize SQLite queries
    - Semaphore to limit concurrent embedding requests
    - Batch embedding API calls
    """
    if not chunks:
        return

    trace_id = get_trace_id()
    from ..embed.cache import get_batch, put_batch

    # Batch cache lookup
    chunk_ids = [c["chunk_id"] for c in chunks]
    cached = get_batch(chunk_ids)

    # Apply cached embeddings
    uncached_chunks = []
    for c in chunks:
        if c["chunk_id"] in cached:
            c["embedding"] = cached[c["chunk_id"]]
        else:
            uncached_chunks.append(c)

    logger.info(f"[{trace_id}] Cache hit: {len(cached)}/{len(chunks)}, embedding {len(uncached_chunks)} chunks")

    if not uncached_chunks:
        return

    # Create batches and embed with backpressure
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
    batches = [uncached_chunks[i:i + EMBED_BATCH_SIZE]
               for i in range(0, len(uncached_chunks), EMBED_BATCH_SIZE)]

    # Run all batches concurrently (semaphore limits actual concurrency)
    tasks = [_embed_batch_async(batch, semaphore) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect new embeddings
    new_embeddings: Dict[str, List[float]] = {}
    for result in results:
        if isinstance(result, dict):
            new_embeddings.update(result)
        elif isinstance(result, Exception):
            logger.warning(f"[{trace_id}] Embedding batch failed: {result}")

    # Apply embeddings to chunks
    for c in uncached_chunks:
        if c["chunk_id"] in new_embeddings:
            c["embedding"] = new_embeddings[c["chunk_id"]]

    # Batch cache write
    put_batch(new_embeddings)
    logger.info(f"[{trace_id}] Embedded and cached {len(new_embeddings)} chunks")


def _embed_in_place(chunks: List[Dict[str, Any]]) -> None:
    """
    Embed chunks in place (synchronous wrapper for async implementation).
    """
    if not chunks:
        return

    # Run async embedding
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create task
            asyncio.create_task(_embed_chunks_async(chunks))
        else:
            loop.run_until_complete(_embed_chunks_async(chunks))
    except RuntimeError:
        # No event loop, create one
        asyncio.run(_embed_chunks_async(chunks))


# ---------- Streaming batch iterator ----------
def _batched(iterable: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield successive batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# ---------- Public entry points ----------
@timed("ingest_folder")
def ingest_folder(driver: Driver, folder: str) -> Tuple[int, int]:
    """
    Parse, chunk, embed, and upsert a folder of documents.

    Uses streaming to limit memory usage: processes STREAM_BATCH_SIZE
    documents at a time.

    Returns:
        (n_docs_ingested, n_chunks_written_with_vectors)
    """
    trace_id = set_trace_id()
    logger.info(f"[{trace_id}] Starting ingestion from {folder}")

    paths = _collect_files(folder)
    if not paths:
        logger.info(f"[{trace_id}] No files found")
        return (0, 0)

    logger.info(f"[{trace_id}] Found {len(paths)} files")

    total_docs = 0
    total_chunks = 0

    # Process in streaming batches to limit memory
    for path_batch in _batched(paths, STREAM_BATCH_SIZE):
        # Parse batch
        docs = _parse_files_parallel(path_batch, workers=PARSE_WORKERS)
        if not docs:
            continue

        # Build chunks
        chunks = _build_chunks(docs)

        # Embed (with async + backpressure)
        _embed_in_place(chunks)

        # Upsert to database
        upsert_docs(driver, docs)
        upsert_chunks(driver, chunks)

        # Count
        batch_chunks_with_vectors = sum(1 for c in chunks if c.get("embedding"))
        total_docs += len(docs)
        total_chunks += batch_chunks_with_vectors

        logger.info(f"[{trace_id}] Batch complete: {len(docs)} docs, {batch_chunks_with_vectors} chunks with vectors")

    logger.info(f"[{trace_id}] Ingestion complete: {total_docs} docs, {total_chunks} chunks")
    return (total_docs, total_chunks)


def ingest_files(driver: Driver, paths: List[str]) -> Tuple[int, int]:
    """
    Ingest specific files.

    Args:
        driver: Neo4j driver
        paths: List of file paths to ingest

    Returns:
        (n_docs_ingested, n_chunks_written_with_vectors)
    """
    trace_id = set_trace_id()
    logger.info(f"[{trace_id}] Ingesting {len(paths)} files")

    docs = _parse_files_parallel(paths, workers=PARSE_WORKERS)
    if not docs:
        return (0, 0)

    chunks = _build_chunks(docs)
    _embed_in_place(chunks)

    upsert_docs(driver, docs)
    upsert_chunks(driver, chunks)

    wrote = sum(1 for c in chunks if c.get("embedding"))
    logger.info(f"[{trace_id}] Ingested {len(docs)} docs, {wrote} chunks with vectors")
    return (len(docs), wrote)
