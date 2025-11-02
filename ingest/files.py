import os
import re
import glob
import json
import hashlib
import chardet
from chardet.universaldetector import UniversalDetector
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import Driver
from ..config import embed_client, EMBED_MODEL
from ..graph.upsert import upsert_docs, upsert_chunks

# ---------- Optional readers ----------
from typing import Any

# pypdf / PyPDF2 fallback — make PdfReader always defined for linters
PdfReader: Any = None
_HAS_PYPDF = False
try:
    from pypdf import PdfReader as _PdfReader  # preferred modern package
    PdfReader = _PdfReader
    _HAS_PYPDF = True
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader  # older package name
        PdfReader = _PdfReader
        _HAS_PYPDF = True
    except Exception:
        PdfReader = None
        _HAS_PYPDF = False

# docx2txt — ensure module name is typed so `docx2txt.process` is valid to the linter
docx2txt: Any = None
_HAS_DOCX2TXT = False
try:
    import docx2txt as _docx2txt
    docx2txt = _docx2txt
    _HAS_DOCX2TXT = True
except Exception:
    docx2txt = None
    _HAS_DOCX2TXT = False

# chardet/universaldetector for naive text reading
UniversalDetector: Any = None
_HAS_CHARDET = False
try:
    from chardet.universaldetector import UniversalDetector as _UniversalDetector
    UniversalDetector = _UniversalDetector
    _HAS_CHARDET = True
except Exception:
    UniversalDetector = None
    _HAS_CHARDET = False

# ---------- Config ----------
_ALLOWED_EXTS = {".pdf",".doc",".docx",".ppt",".pptx",".txt",".md",".html",".htm",".rtf",".csv"}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")   # control chars (except \t \n \r)

PARSE_WORKERS      = int(os.getenv("PARSE_WORKERS", "8"))  # Increased default
EMBED_WORKERS      = int(os.getenv("EMBED_WORKERS", "4"))  # New: parallel embedding
EMBED_MAX_CHARS    = int(os.getenv("EMBED_MAX_CHARS", "12000"))
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1800"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "100"))   # New: batch processing

HEADING_PATTS = [
    re.compile(r"^\s*\d+(\.\d+)+\s+.{2,120}$"),     # e.g., 11.5. Dental Care
    re.compile(r"^\s*#{1,3}\s+.{2,120}$"),          # Markdown headings
    re.compile(r"^\s*[A-Z][A-Z0-9 /&\-]{3,100}\s*$")# ALL CAPS line
]

# ---------- Helpers ----------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _is_corrupted_text(text: str) -> bool:
    """Check if text appears to be corrupted or garbled."""
    if not text or len(text) < 10:
        return False
    
    # Check for excessive repeated characters
    if re.search(r'(.)\1{20,}', text):
        return True
    
    # Check for high ratio of non-printable characters
    printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
    if len(text) > 0 and printable_chars / len(text) < 0.7:
        return True
    
    # Check for excessive Unicode control characters
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    if len(text) > 0 and control_chars / len(text) > 0.1:
        return True
    
    return False

def _sanitize_for_embedding(text: str) -> str:
    if not text:
        return ""
    
    # Check if text is corrupted and skip if so
    if _is_corrupted_text(text):
        print(f"[WARNING] Skipping corrupted text: {text[:100]}...")
        return ""
    
    # Remove control characters (except \t, \n, \r)
    text = _CTRL.sub(" ", text)
    
    # Remove or replace problematic Unicode characters
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Zero-width characters
    text = re.sub(r'[\u2028\u2029]', '\n', text)       # Line/paragraph separators
    text = re.sub(r'[\u00a0]', ' ', text)              # Non-breaking space
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove excessive repeated characters (like "risrisrisris...")
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    
    # Ensure proper encoding
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass
    
    return text.strip()

def _chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sents = re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())
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
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            out.append((tail + " " + c).strip())
    return out

def _read_text_naive(path: str) -> str:
    if _HAS_CHARDET:
        with open(path, "rb") as f:
            raw = f.read()
        enc = (chardet.detect(raw).get("encoding") or "utf-8").strip() or "utf-8"
        try:
            text = raw.decode(enc, errors="ignore")
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    
    # Additional sanitization for text files
    text = _sanitize_for_embedding(text)
    return text

def _extract_text(path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, metadata). Metadata includes source/path/filename/ext.
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
                    # Sanitize PDF text immediately
                    page_text = _sanitize_for_embedding(page_text)
                    pages.append(page_text)
                except Exception:
                    pages.append("")
            return "\n".join(pages), meta

        if ext == ".docx" and _HAS_DOCX2TXT:
            docx_text = docx2txt.process(path) or ""
            docx_text = _sanitize_for_embedding(docx_text)
            return docx_text, meta

        if ext in (".txt", ".md", ".csv", ".rtf", ".html", ".htm"):
            return _read_text_naive(path), meta

        # Fallback (may be empty for binaries but harmless)
        return _read_text_naive(path), meta
    except Exception:
        return "", meta

def _collect_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files: List[str] = []
    for ext in sorted(_ALLOWED_EXTS):
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return files

def _parse_files_parallel(paths: List[str], workers: int = PARSE_WORKERS) -> List[Dict[str, Any]]:
    """
    Returns a list of doc dicts: {"doc_id","title","text","metadata":{...}}
    """
    docs: List[Dict[str, Any]] = []
    if not paths:
        return docs

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_extract_text, p) for p in paths]
        for fut in as_completed(futs):
            text, meta = fut.result()
            if not text or not text.strip():
                continue
            title = os.path.splitext(os.path.basename(meta["filename"]))[0]
            did   = _sha1(title + ":" + text[:2000])
            docs.append({
                "doc_id": did,
                "title": title,
                "text": text,
                "metadata": meta,
            })
    return docs

def _looks_like_heading(line: str) -> bool:
    ln = line.strip()
    if not ln or len(ln) > 140:
        return False
    for rx in HEADING_PATTS:
        if rx.match(ln):
            return True
    # uppercase-ish heuristic
    letters = [c for c in ln if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
        return True
    return False

def _first_heading_in_block(block: str) -> str:
    for line in block.splitlines():
        if _looks_like_heading(line):
            return re.sub(r"\s+", " ", line.strip())
    return ""

def _build_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns list of chunk dicts:
      {"chunk_id","doc_id","order","text","heading","embedding","metadata":{...}}
    """
    out: List[Dict[str, Any]] = []
    for d in docs:
        current_heading = ""
        parts = _chunk_text(d["text"])
        for i, ct in enumerate(parts):
            ct = _sanitize_for_embedding(ct)[:EMBED_MAX_CHARS]
            if not ct:
                continue
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
                "embedding": None,        # will be filled in place
                "metadata": {"from_doc": d["doc_id"]},
            })
    return out

def _embed_batch(batch_chunks: List[Dict[str, Any]]) -> None:
    """Process a batch of chunks for embedding."""
    from ..embed.cache import get as cache_get, put as cache_put
    
    # Check cache first
    uncached_chunks = []
    for c in batch_chunks:
        chunk_id = c["chunk_id"]
        cached_embedding = cache_get(chunk_id)
        if cached_embedding:
            c["embedding"] = cached_embedding
        else:
            uncached_chunks.append(c)
    
    if not uncached_chunks:
        return
    
    # Process uncached chunks in smaller batches
    EMBED_BATCH_SIZE = 32  # Adjust based on your embedding service limits
    for i in range(0, len(uncached_chunks), EMBED_BATCH_SIZE):
        batch = uncached_chunks[i:i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        
        try:
            # Try batch embedding first
            r = embed_client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                encoding_format="float",
            )
            for j, embedding in enumerate(r.data):
                batch[j]["embedding"] = embedding.embedding
                cache_put(batch[j]["chunk_id"], embedding.embedding)
        except Exception:
            # Fallback to individual requests
            for c in batch:
                try:
                    r = embed_client.embeddings.create(
                        model=EMBED_MODEL,
                        input=c["text"],
                        encoding_format="float",
                    )
                    c["embedding"] = r.data[0].embedding
                    cache_put(c["chunk_id"], r.data[0].embedding)
                except Exception:
                    c["embedding"] = None

def _embed_in_place(chunks: List[Dict[str, Any]]) -> None:
    """Parallel embedding processing for better performance."""
    if not chunks:
        return
    
    # Process in parallel batches
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        # Split chunks into batches for parallel processing
        batch_size = max(1, len(chunks) // EMBED_WORKERS)
        futures = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            future = executor.submit(_embed_batch, batch)
            futures.append(future)
        
        # Wait for all batches to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Embedding batch failed: {e}")

# ---------- Public entry point ----------
def ingest_folder(driver: Driver, folder: str) -> Tuple[int, int]:
    """
    Parse, chunk (with headings), embed, and upsert a folder of documents.

    Returns:
      (n_docs_ingested, n_chunks_written_with_vectors)
    """
    paths  = _collect_files(folder)
    docs   = _parse_files_parallel(paths, workers=PARSE_WORKERS)
    if not docs:
        return (0, 0)

    chunks = _build_chunks(docs)
    _embed_in_place(chunks)

    # Batched writes via shared upsert module
    upsert_docs(driver, docs)
    upsert_chunks(driver, chunks)

    wrote = sum(1 for c in chunks if c.get("embedding"))
    return (len(docs), wrote)
