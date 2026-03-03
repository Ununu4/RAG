"""
FastAPI app for RAG QA. Endpoints: /query, /health, /metrics.
Run: uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
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

from chromadb import PersistentClient

from unified_retrieval.rag_qa import answer_query
from unified_retrieval.config import RAGConfig
from unified_retrieval.monitoring import JsonFileStrategy, clear_strategies, register_strategy

app = FastAPI(title="RAG QA API", version="1.0.0")

_DEFAULT_CHROMA = os.getenv("RAG_CHROMA_PATH", str(_ROOT / "chroma_db"))
_METRICS_FILE = _ROOT / "logs" / "rag_metrics.jsonl"

# Persist metrics for every /query (enables /metrics aggregates)
clear_strategies()
register_strategy(JsonFileStrategy(_METRICS_FILE))


def _list_collections():
    client = PersistentClient(path=_DEFAULT_CHROMA)
    return [c.name for c in client.list_collections() if getattr(c, "name", "").startswith("lender-")]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = Field(None, description="Lender name: 'bitty', 'alternative-funding-group', or null for auto-detect")
    tier: str = Field("minimal", description="minimal (fast), balanced, full (best quality)")


class QueryResponse(BaseModel):
    answer: str
    used_sources: int
    collection: str
    run_id: str | None = None
    retrieval_ms: float | None = None
    llm_ms: float | None = None
    total_ms: float | None = None
    faithfulness: float | None = None
    faithfulness_ms: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/collections")
def collections():
    """List available lender collections. Use short names (e.g. 'bitty') in /query."""
    return {"collections": _list_collections()}


def _load_metrics_history(limit: int = 100) -> list[dict]:
    """Read recent metrics from JSONL file."""
    if not _METRICS_FILE.exists():
        return []
    lines = []
    with open(_METRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines[-limit:]


def _percentile(arr: list[float], p: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f]) if f < len(s) - 1 else s[-1]


@app.get("/metrics")
def metrics(limit: int = 100):
    """Aggregated pipeline metrics from recent queries."""
    history = _load_metrics_history(limit)
    if not history:
        return {"count": 0, "message": "No metrics yet. Run /query to populate."}

    total_ms = [h.get("retrieval_ms", 0) + h.get("llm_ms", 0) for h in history if isinstance(h.get("retrieval_ms"), (int, float)) and isinstance(h.get("llm_ms"), (int, float))]
    retrieval_ms = [h["retrieval_ms"] for h in history if isinstance(h.get("retrieval_ms"), (int, float))]
    llm_ms = [h["llm_ms"] for h in history if isinstance(h.get("llm_ms"), (int, float))]
    faithfulness_scores = [h["faithfulness"] for h in history if isinstance(h.get("faithfulness"), (int, float))]

    out = {
        "count": len(history),
        "latency_ms": {
            "total": {"avg": sum(total_ms) / len(total_ms) if total_ms else 0, "p50": _percentile(total_ms, 50), "p95": _percentile(total_ms, 95)},
            "retrieval": {"avg": sum(retrieval_ms) / len(retrieval_ms) if retrieval_ms else 0},
            "llm": {"avg": sum(llm_ms) / len(llm_ms) if llm_ms else 0},
        },
        "recent": history[-10:],
    }
    if faithfulness_scores:
        out["faithfulness"] = {"avg": sum(faithfulness_scores) / len(faithfulness_scores), "count": len(faithfulness_scores)}
    return out


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
            compute_faithfulness=cfg.compute_faithfulness,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    j = out.get("json", {})
    ans = (out.get("answer_text", "") or "").strip()
    try:
        used = int(j.get("used_sources", 0))
    except (TypeError, ValueError):
        used = 0

    m = out.get("metrics")
    run_id = m.run_id if m else None
    collection = m.collection if m else (req.collection or "unknown")
    retrieval_ms = m.retrieval_ms if m else None
    llm_ms = m.llm_ms if m else None
    total_ms = (retrieval_ms + llm_ms) if (retrieval_ms is not None and llm_ms is not None) else None

    faithfulness = m.faithfulness if m else None
    faithfulness_ms = m.faithfulness_ms if (m and m.faithfulness is not None) else None

    return QueryResponse(
        answer=ans,
        used_sources=used,
        collection=collection,
        run_id=run_id,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        faithfulness=faithfulness,
        faithfulness_ms=faithfulness_ms,
    )
