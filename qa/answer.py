import os, math, re
from typing import Any, Dict, List, Tuple

from ..config import embed_client, chat_client, EMBED_MODEL, CHAT_MODEL, NEO4J_DB, mcp_client, MCP_CHAT_MODEL, MCP_AGENT_URL
from .mcp_adapter import call_mcp_agent
from ..graph.query import hybrid_search

# ===== Answer style (generic, extractive) =====
ANSWER_SYSTEM = (
    "You are a precise assistant. Use ONLY the provided context to answer.\n"
    "- Be concise and factual.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Cite supporting chunk(s) inline like [doc:title#chunk_order]."
)

# ===== Packing budgets (tweakable via env) =====
AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "4"))
MAX_CTX_TOKENS     = int(os.getenv("MAX_CTX_TOKENS", "2800"))  # room for prompt + output
MAX_OUTPUT_TOKENS  = int(os.getenv("MAX_OUTPUT_TOKENS", "192"))
PER_CHUNK_CHAR_CAP = int(os.getenv("PER_CHUNK_CHAR_CAP", "900"))
FOCUS_WINDOW       = int(os.getenv("FOCUS_WINDOW", "2"))       # neighbors ±N around best anchor
PER_DOC_MAX        = int(os.getenv("PER_DOC_MAX", "3"))        # max chunks per doc in final context
DOC_ROUND_ROBIN    = os.getenv("DOC_ROUND_ROBIN", "1") == "1"

# ===== Heading-boost parameters =====
STOP = {
    "the","a","an","of","in","to","and","or","for","over","under","within","on","by","with",
    "is","are","was","were","be","been","as","that","this","these","those","at","from","it",
    "its","their","his","her","about","into","out","up","down","not","no","do","does","did"
}
STOP_EXTRA = {t.strip().lower() for t in os.getenv(
    "STOP_EXTRA", "air,force,department,united,states,us,u.s.,guardian,space"
).split(",") if t.strip()}

HEADING_TOKEN_MINLEN = int(os.getenv("HEADING_TOKEN_MINLEN", "5"))
HEADING_UNIT_BOOST   = float(os.getenv("HEADING_UNIT_BOOST", "0.12"))
HEADING_BOOST_CAP    = float(os.getenv("HEADING_BOOST_CAP", "0.45"))
HEADING_PHRASE_BONUS = float(os.getenv("HEADING_PHRASE_BONUS", "0.30"))

# ---------- token/phrase helpers ----------
def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / AVG_CHARS_PER_TOKEN))

def _q_tokens(question: str) -> List[str]:
    q = (question or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-_/]*", q)
    return [t for t in toks
            if len(t) >= HEADING_TOKEN_MINLEN
            and t not in STOP
            and t not in STOP_EXTRA]

def _extract_phrases(question: str) -> List[str]:
    toks = _q_tokens(question)
    phrases = []
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i+1]
        if a != b:
            phrases.append(f"{a} {b}")
    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            out.append(p); seen.add(p)
    return out[:5]

# ---------- context packing ----------
def _build_context_block_fit_diverse(chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    # group by doc_id
    doc_order, by_doc = [], {}
    for ch in chunks:
        d = ch["doc_id"]
        if d not in by_doc:
            by_doc[d] = []
            doc_order.append(d)
        by_doc[d].append(ch)

    def iterator():
        if not DOC_ROUND_ROBIN:
            for d in doc_order:
                for ch in by_doc[d][:PER_DOC_MAX]:
                    yield ch
        else:
            lanes = [by_doc[d][:PER_DOC_MAX] for d in doc_order]
            i = 0
            while True:
                emitted = False
                for lane in lanes:
                    if i < len(lane):
                        yield lane[i]
                        emitted = True
                if not emitted:
                    break
                i += 1

    lines, cits, used = [], [], 0
    for ch in iterator():
        tag  = f"[{ch['doc_id'][:8]}:{(ch.get('title') or 'Untitled')}#{ch['order']}]"
        text = (ch.get("text") or "")[:PER_CHUNK_CHAR_CAP]
        block = f"{tag}\n{text}\n\n"
        tks = _approx_tokens(block)
        if used + tks > MAX_CTX_TOKENS:
            break
        lines.append(block); cits.append(tag); used += tks
    return "".join(lines), cits

# ---------- FULLTEXT helpers ----------
def _fts_query_from_question(q: str) -> str:
    toks    = _q_tokens(q)
    phrases = _extract_phrases(q)
    parts = []
    parts += [f"\"{p}\"" for p in phrases]
    parts += toks
    return " OR ".join(parts) if parts else (q or "")

def _fts_best_chunk(driver, query_str: str):
    if not query_str:
        return None
    from neo4j import RoutingControl  # local import to avoid global dep
    with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
        rows = s.run("""
            CALL db.index.fulltext.queryNodes('chunk_text_fts', $q, {limit: 1})
            YIELD node, score
            RETURN node.doc_id AS doc_id, node.`order` AS ord
        """, q=query_str).data()
    return (rows[0]["doc_id"], rows[0]["ord"]) if rows else None

def _neighbor_window(driver, doc_id: str, center_ord: int, window: int) -> List[Dict[str, Any]]:
    from neo4j import RoutingControl
    lo, hi = max(0, center_ord - window), center_ord + window
    with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
        rows = s.run("""
            MATCH (d:Document {doc_id:$doc})-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.`order` >= $lo AND c.`order` <= $hi
            RETURN c.chunk_id AS chunk_id, c.text AS text, c.heading AS heading, c.`order` AS `order`,
                   c.doc_id AS doc_id, d.title AS title
            ORDER BY c.`order`
        """, doc=doc_id, lo=lo, hi=hi).data()
    out = []
    for r in rows:
        out.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "heading": r.get("heading"),
            "order": r["order"],
            "doc_id": r["doc_id"],
            "title": r["title"] or ""
        })
    return out

# ---------- heading hydration + boost ----------
def _hydrate_headings(driver, chunks: List[Dict[str, Any]]) -> None:
    need = [c for c in chunks if not c.get("heading")]
    if not need:
        return
    from neo4j import RoutingControl
    ids = [c["chunk_id"] for c in need]
    with driver.session(database=NEO4J_DB, default_access_mode=RoutingControl.READ) as s:
        rows = s.run("""
            MATCH (c:Chunk) WHERE c.chunk_id IN $ids
            RETURN c.chunk_id AS chunk_id, c.heading AS heading
        """, ids=ids).data()
    hmap = {r["chunk_id"]: (r.get("heading") or "") for r in rows}
    for c in chunks:
        if not c.get("heading"):
            h = hmap.get(c["chunk_id"], "")
            if not h:
                t = (c.get("text") or "").strip()
                h = t.split("\n", 1)[0].strip()
                if len(h) < 12:
                    m = re.split(r"(?<=[\.\!\?])\s+", t, maxsplit=1)
                    h = (m[0] if m else t)[:120].strip()
            c["heading"] = h

def _apply_heading_boost(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qwords  = set(_q_tokens(question))
    phrases = [p.lower() for p in _extract_phrases(question)]
    if not qwords and not phrases:
        return chunks

    for c in chunks:
        if "heading" not in c or c["heading"] is None:
            c["heading"] = ""

    scored = []
    for idx, c in enumerate(chunks):
        h = (c.get("heading") or "").lower()
        hits = sum(1 for w in qwords if w in h)
        boost = min(HEADING_BOOST_CAP, HEADING_UNIT_BOOST * hits) if hits else 0.0
        if boost < HEADING_BOOST_CAP and any(p in h for p in phrases):
            boost = min(HEADING_BOOST_CAP, boost + HEADING_PHRASE_BONUS)
        scored.append((boost, idx, c))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored]

# ---------- Main entry point ----------
def ask(driver, question: str, top_k: int = 6, use_mcp: bool = False) -> Dict[str, Any]:
    # 1) Embed the question
    q_emb = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=question,
        encoding_format="float"
    ).data[0].embedding

    # 2) Prepared FTS query (tokens + quoted bigrams)
    fts_q = _fts_query_from_question(question)

    # 3) Anchor on best FTS hit + neighbor window
    ctx_chunks: List[Dict[str, Any]] = []
    best = _fts_best_chunk(driver, fts_q)
    if best:
        doc_id, ord0 = best
        ctx_chunks = _neighbor_window(driver, doc_id, ord0, FOCUS_WINDOW)

    # 4) Supplement with hybrid (pass prepared FTS query)
    try:
        hyb = hybrid_search(driver, q_emb, fts_q, top_k=top_k) or []
    except Exception as e:
        # Optional: print or log for debugging
        # print(f"[hybrid_search error] {e}")
        hyb = []

    seen = {c["chunk_id"] for c in ctx_chunks}
    for h in hyb:
        if h["chunk_id"] not in seen:
            h.setdefault("heading", None)
            ctx_chunks.append(h)
            seen.add(h["chunk_id"])

    # 5) Hydrate headings and apply boost
    _hydrate_headings(driver, ctx_chunks)
    ctx_chunks = _apply_heading_boost(question, ctx_chunks)

    # 6) Pack + ask
    context_text, citations = _build_context_block_fit_diverse(ctx_chunks)
    user_prompt = (
        "Answer the question using ONLY the context below. "
        "Be concise, list key facts, and include inline citations.\n\n"
        f"Question:\n{question}\n\nContext:\n{context_text}"
    ).strip()

    # Initialize variables
    resp = None
    used_mcp = False
    fallback_error = None
    agent_err = None

    # If use_mcp is requested, prefer agent gateway (MCP_AGENT_URL) because the Learn MCP
    # server is intended to be used via an agent framework. If MCP_AGENT_URL isn't set,
    # fall back to attempting the raw mcp_client (best-effort) and then the default chat client.
    if use_mcp:
        # 1) Agent gateway (preferred)
        if MCP_AGENT_URL:
            msgs = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user_prompt}]
            ok, content_or_err = call_mcp_agent(msgs, agent_url=MCP_AGENT_URL)
            if ok:
                # Create a simple response object
                class SimpleResponse:
                    def __init__(self, content):
                        self.choices = [SimpleChoice(content)]
                
                class SimpleChoice:
                    def __init__(self, content):
                        self.message = SimpleMessage(content)
                
                class SimpleMessage:
                    def __init__(self, content):
                        self.content = content
                
                resp = SimpleResponse(content_or_err)
                used_mcp = True
            else:
                # Agent gateway didn't work; fall through to mcp_client/raw attempt
                agent_err = content_or_err
                resp = None
                used_mcp = False

        # 2) If we didn't get a response from agent gateway, try raw mcp_client (best-effort)
        if (not use_mcp) or (use_mcp and (not MCP_AGENT_URL or resp is None)):
            try:
                client = mcp_client if use_mcp else chat_client
                model = MCP_CHAT_MODEL if use_mcp else CHAT_MODEL
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user",   "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=MAX_OUTPUT_TOKENS
                )
                used_mcp = use_mcp
            except Exception as e:
                # If we tried mcp_client and failed, attempt the default chat client
                if use_mcp:
                    try:
                        resp = chat_client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": ANSWER_SYSTEM},
                                {"role": "user",   "content": user_prompt}
                            ],
                            temperature=0.0,
                            max_tokens=MAX_OUTPUT_TOKENS
                        )
                        used_mcp = False
                        fallback_error = e
                    except Exception as ee:
                        err_msg = f"Agent gateway error: {agent_err or '<no agent_err>'}; mcp_client error: {repr(e)}; fallback error: {repr(ee)}"
                        return {"answer": "", "citations": [], "used_chunks": ctx_chunks, "error": err_msg}
                else:
                    # we were not using MCP and the default chat client failed
                    return {"answer": "", "citations": [], "used_chunks": ctx_chunks, "error": f"Chat client error: {repr(e)}"}
    else:
        # Not using MCP: normal default chat_client path
        try:
            resp = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS
            )
            used_mcp = False
        except Exception as e:
            return {"answer": "", "citations": [], "used_chunks": ctx_chunks, "error": f"Chat client error: {repr(e)}"}

    # safe extraction: handle missing fields / None content
    content = ""
    if resp:
        try:
            # prefer getattr to avoid AttributeError if structure differs
            content = getattr(resp.choices[0].message, "content", None) or ""
        except Exception:
            # if you want diagnostics, import logging and use logging.exception(...)
            content = ""

    # Sanitize the answer content
    answer = content.strip()
    
    # Clean up any encoding issues in the answer
    if answer:
        # Remove problematic Unicode characters
        answer = re.sub(r'[\u200b-\u200d\ufeff]', '', answer)  # Zero-width characters
        answer = re.sub(r'[\u2028\u2029]', '\n', answer)       # Line/paragraph separators
        answer = re.sub(r'[\u00a0]', ' ', answer)              # Non-breaking space
        
        # Remove excessive repeated characters
        answer = re.sub(r'(.)\1{10,}', r'\1', answer)
        
        # Normalize whitespace
        answer = re.sub(r'\n+', '\n', answer)
        answer = re.sub(r'[ \t]+', ' ', answer)
        
        # Ensure proper encoding
        try:
            answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            pass
    
    # If we fell back from MCP, append a short note (non-intrusive)
    if not used_mcp and fallback_error:
        note = "\n\n[Note: MCP request failed and response was served from the default chat backend.]"
        answer = answer + note

    return {"answer": answer, "citations": citations, "used_chunks": ctx_chunks}
