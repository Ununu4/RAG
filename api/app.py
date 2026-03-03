"""
FastAPI app for RAG QA. Endpoints: /query, /health, /metrics.
Run: uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root and unified_retrieval on path (for rag_qa's query_improved, backends, etc.)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "unified_retrieval") not in sys.path:
    sys.path.insert(0, str(_ROOT / "unified_retrieval"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from unified_retrieval.rag_qa import answer_query
from unified_retrieval.config import RAGConfig

app = FastAPI(title="RAG QA API", version="1.0.0")

_DEFAULT_CHROMA = os.getenv("RAG_CHROMA_PATH", str(_ROOT / "chroma_db"))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = None
    tier: str = "balanced"


class QueryResponse(BaseModel):
    answer: str
    used_sources: int
    coherence: float | None
    coherence_supported: int | None
    coherence_total: int | None
    collection: str
    run_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Placeholder for pipeline metrics. Extend with JsonFileStrategy if needed."""
    return {"message": "Enable --metrics in CLI for JSONL output"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    cfg = RAGConfig.from_env()
    if req.tier:
        cfg.tier = req.tier
        cfg._apply_tier()

    try:
        out = answer_query(
            req.query,
            req.collection,
            _DEFAULT_CHROMA,
            n_results=cfg.n_results,
            expand_neighbors=cfg.expand_neighbors,
            use_rerank=cfg.use_rerank,
            max_chars_per_doc=cfg.max_chars_per_doc,
            include_collection_context=True,
            collection_max_docs=150,
            collection_chars=6000,
            num_ctx=cfg.num_ctx,
            num_predict=cfg.num_predict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    j = out.get("json", {})
    ans = (out.get("answer_text", "") or "").strip()
    try:
        used = int(j.get("used_sources", 0))
    except (TypeError, ValueError):
        used = 0

    coh = j.get("coherence")
    cs = j.get("coherence_supported")
    ct = j.get("coherence_total")
    if isinstance(coh, (int, float)) and isinstance(cs, int) and isinstance(ct, int) and ct > 0:
        coh_val = float(coh)
    else:
        coh_val = None
        cs = None
        ct = None

    m = out.get("metrics")
    run_id = m.run_id if m else None
    collection = m.collection if m else (req.collection or "unknown")

    return QueryResponse(
        answer=ans,
        used_sources=used,
        coherence=coh_val,
        coherence_supported=cs,
        coherence_total=ct,
        collection=collection,
        run_id=run_id,
    )
