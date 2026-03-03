# rag_qa.py
from __future__ import annotations
import json
import logging
import re
import textwrap
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from query_improved import semantic_query

from monitoring import (
    CostAwareStrategy,
    JsonFileStrategy,
    LoggingStrategy,
    PipelineMetrics,
    clear_strategies,
    get_logger,
    notify,
    register_strategy,
    setup_logging,
    timed,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "nous-hermes2:latest"

def _format_sources(results: Dict, max_chars_per_doc: int = 2000) -> str:
    lines = []
    for i, (rid, doc, meta) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0]), 1):
        lender = meta.get("lender_name", "?")
        section = meta.get("section", "")
        src = f"[S{i}] id={rid} | lender={lender} | section={section}"
        preview = doc if len(doc) <= max_chars_per_doc else doc[:max_chars_per_doc] + "..."
        lines.append(src + "\n" + preview)
    return "\n\n".join(lines)

def _build_messages(query: str, sources_block: str, json_schema_hint: str, background: Optional[str] = None) -> List[Dict]:
    system = (
        "You are a careful financial analyst. Answer ONLY from provided sources. "
        "Cite multiple sources using their bracket IDs like [S1], [S2]. If unsure, say so. "
        "Avoid repetition and spelling mistakes."
    )
    # We intentionally omit background and distilled bullets to avoid content mixing
    bg = ""
    user = f"""
Task:
- Answer comprehensively and precisely using ONLY the Sources block below.
- Do NOT include citations or [S#] IDs in the answer text.
- Use clear sections and bullet lists when listing industries, states, documents, or notes.
- Include 2-5 actionable recommendations ONLY if grounded in the sources.
- Output strictly valid JSON matching the schema. No extra keys. No prose outside JSON.

Use this exact section order and headings (omit sections with no data):
1) Eligibility thresholds
2) NSFs/Negative days
3) Positions funded
4) Term ranges
5) Rate/Fee structure
6) Max exposure
7) Prepayment/Early payoff
8) Renewal/Early renewal policy
9) Stacking/Refi rules
10) State/Industry restrictions
11) Required documents by tiers
12) Recommendations

Formatting rules:
- Use simple "- " bullets for lists (dash + space)
- Avoid repetition and filler; be direct and factual
- Make every sentence attributable to the Sources content
- Ignore any mention of other lenders; apply only the identified lender's policies

User query:
{query}

Sources:
{sources_block}

JSON schema (exact):
{json_schema_hint}
{bg}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": textwrap.dedent(user).strip()},
    ]

def _distill_sources(results: Dict, max_bullets: int = 60) -> str:
    """Extract salient bullet lines from retrieved documents for dense guidance."""
    bullets: List[str] = []
    docs = results.get("documents", [[]])[0] if results else []
    for doc in docs:
        for line in (doc or "").splitlines():
            s = line.strip()
            if (s.startswith("*") or s.startswith("-")) and len(s) > 3:
                bullets.append(s.lstrip("*- ").strip())
    if len(bullets) > max_bullets:
        bullets = bullets[:max_bullets]
    return "\n".join(f"- {b}" for b in bullets)

def _extract_citations(text: str) -> List[str]:
    seen = set()
    cites: List[str] = []
    for m in re.finditer(r"\[S(\d+)\]", text or ""):
        sid = f"S{m.group(1)}"
        if sid not in seen:
            seen.add(sid)
            cites.append(sid)
    return cites

def _build_source_index(results: Dict) -> List[Dict]:
    index: List[Dict] = []
    ids = results.get("ids", [[]])[0] if results else []
    mets = results.get("metadatas", [[]])[0] if results else []
    for i, (rid, meta) in enumerate(zip(ids, mets), 1):
        lender = meta.get("lender_name", "?") if isinstance(meta, dict) else "?"
        section = meta.get("section", "") if isinstance(meta, dict) else ""
        index.append({"sid": f"S{i}", "id": rid, "lender": lender, "section": section})
    return index

def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _get_lender_tokens(chroma_path: str) -> set[str]:
    client = PersistentClient(path=chroma_path)
    tokens: set[str] = set()
    for c in client.list_collections():
        name = getattr(c, "name", "")
        if not name.startswith("lender-"):
            continue
        slug = name.replace("lender-", "")
        tokens.add(_normalize_token(slug))
        try:
            col = client.get_collection(name)
            meta = getattr(col, "metadata", {}) or {}
            lender_name = _normalize_token(meta.get("lender_name", ""))
            if lender_name:
                tokens.add(lender_name)
        except Exception:
            pass
    return {t for t in tokens if len(t) >= 3}

def _filter_cross_lender_mentions(results: Dict, expected_slug: str, chroma_path: str) -> Dict:
    all_tokens = _get_lender_tokens(chroma_path)
    exp = _normalize_token(expected_slug)
    other = {t for t in all_tokens if t and t != exp and exp not in t}

    ids = results.get("ids", [[]])[0] if results else []
    docs = results.get("documents", [[]])[0] if results else []
    metas = results.get("metadatas", [[]])[0] if results else []

    keep_ids, keep_docs, keep_metas = [], [], []
    for rid, d, m in zip(ids, docs, metas):
        txt = f" {_normalize_token(d)} "
        if any(f" {tok} " in txt for tok in other):
            continue
        keep_ids.append(rid); keep_docs.append(d); keep_metas.append(m)

    if keep_ids:
        return {"ids": [keep_ids], "documents": [keep_docs], "metadatas": [keep_metas]}
    return results

def _filter_results_by_lender(results: Dict, expected_slug: str) -> Dict:
    def _to_slug(s: str) -> str:
        return re.sub(r"[^\w\-]+", "-", (s or "").lower()).strip("-")
    ids = results.get("ids", [[]])[0] if results else []
    docs = results.get("documents", [[]])[0] if results else []
    metas = results.get("metadatas", [[]])[0] if results else []
    keep_ids, keep_docs, keep_metas = [], [], []
    for rid, d, m in zip(ids, docs, metas):
        lender_name = m.get("lender_name") if isinstance(m, dict) else ""
        if _to_slug(lender_name) == expected_slug:
            keep_ids.append(rid)
            keep_docs.append(d)
            keep_metas.append(m)
    if keep_ids:
        return {"ids": [keep_ids], "documents": [keep_docs], "metadatas": [keep_metas]}
    return results

_ST: Optional[SentenceTransformer] = None
def _get_st() -> SentenceTransformer:
    global _ST
    if _ST is None:
        _ST = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    return _ST

def _split_answer_bullets(answer_text: str) -> List[str]:
    lines: List[str] = []
    for ln in (answer_text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("-", "*")) or ":" in s or len(s) > 20:
            lines.append(s.lstrip("-* ").strip())
    return lines[:50]

def _coherence_score(answer_text: str, results: Dict, thresh: float = 0.55) -> Tuple[float, Tuple[int, int]]:
    bullets = _split_answer_bullets(answer_text)
    if not bullets:
        return 0.0, (0, 0)
    docs = results.get("documents", [[]])[0] if results else []
    if not docs:
        return 0.0, (0, len(bullets))
    st = _get_st()
    q_embs = st.encode(bullets)
    d_texts = [d[:1000] for d in docs]
    d_embs = st.encode(d_texts)
    d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-12)
    sims: List[float] = []
    for q in q_embs:
        qn = q / (np.linalg.norm(q) + 1e-12)
        scores = d_norm @ qn
        sims.append(float(np.max(scores)))
    avg = float(np.mean(sims))
    supported = sum(1 for s in sims if s >= thresh)
    return avg, (supported, len(bullets))

def _polish_answer(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return a
    cleaned: List[str] = []
    seen: set = set()
    for ln in a.splitlines():
        s = ln.rstrip()
        # drop stray JSON echoes
        if s.startswith(('"answer"', "'answer'", '{', '}')):
            continue
        if s in seen:
            continue
        seen.add(s)
        # normalize bullets
        if s.startswith(("• ", "– ", "* ")):
            s = "- " + s[2:].strip()
        cleaned.append(s)
    # condense blank lines
    out: List[str] = []
    prev_blank = False
    for s in cleaned:
        if not s.strip():
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        out.append(s)
    return "\n".join(out).strip()

def _split_list_str(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    parts = re.split(r",|;| and ", s, flags=re.I)
    items = [re.sub(r"\s+", " ", p).strip(" .") for p in parts]
    return [i for i in items if i]

def _normalize_to_schema(obj: Dict) -> Dict:
    norm: Dict = {
        "answer": (obj.get("answer") or ""),
        "details": obj.get("details") if isinstance(obj.get("details"), list) else [],
        "requirements": {
            "fico_min": None,
            "revenue_min": None,
            "tib_min": None,
            "restricted_states": [],
            "restricted_industries": [],
            "required_documents": [],
        },
        "programs": [],
        "exclusions": [],
        "notes": [],
        "recommendations": obj.get("recommendations") if isinstance(obj.get("recommendations"), list) else [],
        "sources": obj.get("sources") if isinstance(obj.get("sources"), list) else [],
    }

    # Map model-specific keys to canonical schema
    hri = obj.get("high_risk_industries") or {}
    if isinstance(hri, dict):
        for k, v in hri.items():
            lst = _split_list_str(v)
            if not lst:
                continue
            kl = (k or "").lower()
            if "prohibited" in kl or "not_for_profit" in kl or "not_for_profits" in kl:
                norm["exclusions"].extend(lst)
            else:
                norm["requirements"]["restricted_industries"].extend(lst)

    # Submission criteria -> required docs
    ksc = obj.get("key_submission_criteria")
    if isinstance(ksc, str) and "tax return" in ksc.lower():
        norm["requirements"]["required_documents"].append("Tax returns (depends on amount)")

    # Popular industries -> notes
    pop = obj.get("popular_industries")
    if isinstance(pop, str):
        norm["notes"].append(f"Popular industries: {pop.strip()}")

    # Deduplicate lists
    for k in ("restricted_states", "restricted_industries", "required_documents"):
        items = norm["requirements"][k]
        seen: set = set()
        cleaned: List[str] = []
        for it in items:
            it2 = (it or "").strip()
            key = it2.lower()
            if it2 and key not in seen:
                seen.add(key)
                cleaned.append(it2)
        norm["requirements"][k] = cleaned

    for k in ("programs", "exclusions", "notes", "recommendations"):
        seq = norm.get(k) or []
        seen: set = set()
        cleaned: List[str] = []
        for it in seq:
            it2 = it.strip() if isinstance(it, str) else it
            key = it2.lower() if isinstance(it2, str) else None
            if isinstance(it2, str) and it2 and key not in seen:
                seen.add(key)
                cleaned.append(it2)
        norm[k] = cleaned

    # Ensure an answer summary exists
    if not norm["answer"]:
        ri = norm["requirements"]["restricted_industries"]
        if ri:
            sample = ", ".join(ri[:6]) + ("…" if len(ri) > 6 else "")
            norm["answer"] = f"Restricted industries identified: {sample}."
        else:
            norm["answer"] = "Relevant restrictions and requirements identified from sources."

    return norm

def _render_answer_text(obj: Dict) -> str:
    """Render a clean, human-readable answer from structured JSON fields."""
    lines: List[str] = []
    ans = obj.get("answer") or obj.get("answer_text")
    if isinstance(ans, str) and ans.strip():
        lines.append(ans.strip())

    req = obj.get("requirements", {}) or {}
    ri = req.get("restricted_industries") or []
    if isinstance(ri, list) and ri:
        lines.append("\nRestricted industries:")
        for item in ri:
            lines.append(f"- {item}")

    rs = req.get("restricted_states") or []
    if isinstance(rs, list) and rs:
        lines.append("\nRestricted states:")
        for st in rs:
            lines.append(f"- {st}")

    docs = req.get("required_documents") or []
    if isinstance(docs, list) and docs:
        lines.append("\nRequired documents:")
        for d in docs:
            lines.append(f"- {d}")

    # Include programs/exclusions if present
    programs = obj.get("programs") or []
    if isinstance(programs, list) and programs:
        lines.append("\nPrograms:")
        for p in programs:
            lines.append(f"- {p}")

    exclusions = obj.get("exclusions") or []
    if isinstance(exclusions, list) and exclusions:
        lines.append("\nExclusions:")
        for e in exclusions:
            lines.append(f"- {e}")

    notes = obj.get("notes") or []
    if isinstance(notes, list) and notes:
        lines.append("\nNotes:")
        for n in notes:
            lines.append(f"- {n}")

    recs = obj.get("recommendations") or []
    if isinstance(recs, list) and recs:
        lines.append("\nRecommendations:")
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()

def ask_ollama(
    messages: List[Dict],
    temperature: float = 0.1,
    num_ctx: int = 12288,
    num_predict: int = 512,
) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        "stream": False,
        "format": "json",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama chat returns {'message': {'content': '...'}}; extract text
    return data["message"]["content"]

def _slugify(text: str) -> str:
    return re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")


def _detect_collection_for_query(query: str, chroma_path: str, default_collection: str) -> str:
    client = PersistentClient(path=chroma_path)
    cols = client.list_collections()

    # Tokenize query slug and remove generic lender terms
    q_slug = _slugify(query)
    q_tokens = [t for t in q_slug.split("-") if len(t) >= 3]
    stop = {
        "funding", "group", "capital", "financial", "finance", "platform",
        "solutions", "advance", "advances", "business", "credit", "loans", "loan",
    }
    q_tokens = [t for t in q_tokens if t not in stop]
    q_set = set(q_tokens)

    best_score = 0
    best_name = default_collection

    for c in cols:
        name = getattr(c, "name", "")
        if not name.startswith("lender-"):
            continue
        slug = name.replace("lender-", "")
        s_tokens = [t for t in slug.split("-") if t and t not in stop]
        s_set = set(s_tokens)

        # Primary score: token overlap
        overlap = len(q_set & s_set)

        # Secondary: partial containment (e.g., "aspire" matches "aspire-funding-platform")
        if overlap == 0 and any(qt in st for qt in q_set for st in s_set):
            overlap = 1

        if overlap > best_score:
            best_score = overlap
            best_name = name

    return best_name if best_score > 0 else default_collection


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(0, len((text or "").strip()) // 4)


def answer_query(
    query: str,
    collection: Optional[str],
    chroma_path: str,
    n_results: int = 6,
    expand_neighbors: int = 1,
    use_rerank: bool = False,
    max_chars_per_doc: int = 2000,
    include_collection_context: bool = True,
    collection_max_docs: int = 150,
    collection_chars: int = 6000,
    num_ctx: int = 12288,
    num_predict: int = 512,
    metrics: Optional[PipelineMetrics] = None,
) -> Dict:
    m = metrics or PipelineMetrics()
    m.query = query
    m.run_id = str(uuid.uuid4())[:8]

    # 1) Retrieve top evidence (MMR + rerank + optional neighbor expansion)
    chosen_collection = collection or _detect_collection_for_query(
        query, chroma_path, default_collection="lender-alternative-funding-group"
    )
    m.collection = chosen_collection

    def on_retrieval(stats: Dict) -> None:
        m.retrieval_candidates = stats.get("retrieval_candidates", 0)
        m.retrieval_after_mmr = stats.get("retrieval_after_mmr", 0)
        m.retrieval_after_rerank = stats.get("retrieval_after_rerank", 0)
        m.retrieval_after_expand = stats.get("retrieval_after_expand", 0)
        m.retrieval_ms = stats.get("retrieval_ms", 0)
        m.embedding_calls = stats.get("embedding_calls", 0)
        m.cross_encoder_calls = stats.get("cross_encoder_calls", 0)

    results = semantic_query(
        query_text=query,
        collection_name=chosen_collection,
        chroma_path=chroma_path,
        n_results=n_results,
        mmr=True,
        rerank=use_rerank,
        expand_neighbors=expand_neighbors,
        metrics_callback=on_retrieval,
    )

    # Strictly keep docs from the identified lender only
    expected_slug = chosen_collection.replace("lender-", "")
    results = _filter_results_by_lender(results, expected_slug)
    # Drop docs that mention other lenders at the content level
    results = _filter_cross_lender_mentions(results, expected_slug, chroma_path)
    m.retrieval_after_filter = len(results.get("ids", [[]])[0]) if results else 0
    notify("on_retrieval_end", m)
    # If nothing survives filtering, fail fast with a grounded message
    if not results.get("ids") or not results["ids"][0]:
        m.error = "no_results_after_filter"
        notify("on_pipeline_end", m)
        return {
            "json": {"used_sources": 0, "coherence": 0.0},
            "answer_text": f"No lender-specific results found for '{expected_slug}'. Please verify the lender name or try a different query.",
            "metrics": m,
        }

    # 2) Compact sources for prompt
    sources_block = _format_sources(results, max_chars_per_doc=max_chars_per_doc)

    # Background and distilled bullets intentionally disabled to avoid mixing
    background = None

    # 3) Ask LLM for minimal structured JSON (answer + used_sources)
    json_schema_hint = """{
  "answer": "human-readable answer with sections and bullet lists as needed; no citations",
  "used_sources": 0
}"""
    messages = _build_messages(query, sources_block, json_schema_hint, background=background)
    prompt_text = " ".join(str(m.get("content", "")) for m in messages)
    m.prompt_tokens_approx = _approx_tokens(prompt_text)
    notify("on_llm_start", m)

    try:
        t0 = time.perf_counter()
        raw = ask_ollama(messages, num_ctx=num_ctx, num_predict=num_predict)
        m.llm_ms = (time.perf_counter() - t0) * 1000
        m.completion_tokens_approx = _approx_tokens(raw)
    except requests.HTTPError as e:
        # Retry on server error with trimmed prompt/context
        status = getattr(e.response, "status_code", None)
        if status == 500:
            try:
                trimmed_doc_chars = max(800, int(max_chars_per_doc * 0.6))
                trimmed_coll_chars = max(3000, int(collection_chars * 0.6))
                trimmed_ctx = min(num_ctx, 8192)

                # Rebuild prompt smaller
                sources_block = _format_sources(results, max_chars_per_doc=trimmed_doc_chars)
                background = None
                if include_collection_context:
                    try:
                        client_bg = PersistentClient(path=chroma_path)
                        col_bg = client_bg.get_collection(chosen_collection)
                        got = col_bg.get(limit=collection_max_docs)
                        docs_bg = got.get("documents", [])
                        metas_bg = got.get("metadatas", [])
                        acc: List[str] = []
                        total = 0
                        for d, m in zip(docs_bg, metas_bg):
                            section = m.get("section") if isinstance(m, dict) else None
                            prefix = f"[{section}] " if section else ""
                            snippet = (d or "")[:300]
                            piece = prefix + snippet
                            if total + len(piece) > trimmed_coll_chars:
                                break
                            acc.append(piece)
                            total += len(piece)
                        background = "\n".join(acc)
                    except Exception:
                        background = None

                messages = _build_messages(query, sources_block, json_schema_hint, background=background)
                m.llm_retries += 1
                t0 = time.perf_counter()
                raw = ask_ollama(messages, num_ctx=trimmed_ctx, num_predict=num_predict)
                m.llm_ms += (time.perf_counter() - t0) * 1000
                m.completion_tokens_approx = _approx_tokens(raw)
            except Exception:
                raise
        else:
            raise

    notify("on_llm_end", m)

    # 4) Parse JSON robustly (fallback: extract first JSON block)
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {"answer": raw, "used_sources": 0}
        else:
            obj = {"answer": raw, "used_sources": 0}

    # 5) Determine sources count (prefer model's used_sources; fallback to retrieved count)
    used_sources = obj.get("used_sources")
    if not isinstance(used_sources, int) or used_sources <= 0:
        total = len(results.get("ids", [[]])[0]) if results else 0
        used_sources = min(total, n_results)
        obj["used_sources"] = used_sources

    # Render final answer text directly from the model's answer (no normalization to avoid mixing)
    answer_text = (obj.get("answer") or "").strip()
    if not answer_text:
        answer_text = "No summary produced from sources."

    # Compute coherence score against retrieved docs
    coh, (hit, total) = _coherence_score(answer_text, results)
    obj["coherence"] = coh
    obj["coherence_supported"] = hit
    obj["coherence_total"] = total

    m.coherence_score = coh
    m.coherence_supported = hit
    m.coherence_total = total
    m.sources_used = used_sources
    m.answer_length = len(answer_text)
    notify("on_pipeline_end", m)

    return {"json": obj, "answer_text": answer_text, "metrics": m}

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_CHROMA = str(_ROOT / "chroma_db")

    p = argparse.ArgumentParser(description="RAG -> Ollama structured answer")
    p.add_argument("query")
    p.add_argument("--collection", default=None)
    p.add_argument("--chroma", default=_DEFAULT_CHROMA, help="Chroma DB path")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--rerank", action="store_true", default=False)
    p.add_argument("--doc-chars", type=int, default=2000, help="Max chars per source doc in prompt")
    p.add_argument("--num-ctx", type=int, default=12288, help="Ollama context window")
    p.add_argument("--num-predict", type=int, default=512, help="Max tokens to generate")
    p.add_argument("--with-collection", action="store_true", default=True, help="Include lender-wide background context")
    p.add_argument("--coll-maxdocs", type=int, default=150, help="Max docs sampled from collection for background")
    p.add_argument("--coll-chars", type=int, default=6000, help="Max total chars of background context")
    p.add_argument("--metrics", action="store_true", help="Enable logging + JSONL metrics output")
    p.add_argument("--log-dir", type=Path, default=_ROOT / "logs", help="Log directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--tier", choices=["minimal", "balanced", "full"], default="balanced",
                    help="Cost/performance tier: minimal (fast), balanced, full (best quality)")
    args = p.parse_args()

    if args.tier == "minimal":
        args.n = 3
        args.expand = 0
        args.rerank = False
        args.doc_chars = 1200
        args.num_ctx = 8192
        args.num_predict = 256
    elif args.tier == "full":
        args.n = 8
        args.expand = 2
        args.rerank = True
        args.doc_chars = 2500
        args.num_ctx = 16384
        args.num_predict = 768

    if args.metrics:
        clear_strategies()
        setup_logging(
            level=getattr(logging, args.log_level),
            log_dir=args.log_dir,
            log_file="rag_pipeline.log",
            console=True,
        )
        register_strategy(LoggingStrategy())
        register_strategy(JsonFileStrategy(args.log_dir / "rag_metrics.jsonl"))
        register_strategy(CostAwareStrategy())

    out = answer_query(
        args.query,
        args.collection,
        args.chroma,
        n_results=args.n,
        expand_neighbors=args.expand,
        use_rerank=args.rerank,
        max_chars_per_doc=args.doc_chars,
        include_collection_context=args.with_collection,
        collection_max_docs=args.coll_maxdocs,
        collection_chars=args.coll_chars,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    # Print count and answer
    used = out.get("json", {}).get("used_sources")
    try:
        used_int = int(used) if used is not None else 0
    except Exception:
        used_int = 0
    print(f"Sources used: {used_int}")
    coh = out.get("json", {}).get("coherence")
    cs = out.get("json", {}).get("coherence_supported")
    ct = out.get("json", {}).get("coherence_total")
    if isinstance(coh, float) and isinstance(cs, int) and isinstance(ct, int) and ct > 0:
        print(f"Coherence: {coh:.2f} ({cs}/{ct})")
    if args.metrics and "metrics" in out:
        m = out["metrics"]
        print(f"Run: {m.run_id} | retrieval_ms={m.retrieval_ms:.0f} llm_ms={m.llm_ms:.0f} tokens~{m.prompt_tokens_approx}+{m.completion_tokens_approx}")
    print()
    ans_text = (out.get("answer_text", "") or "").strip()
    ans_text = _polish_answer(ans_text)
    print(ans_text)