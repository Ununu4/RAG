# RAG — Lender Recommendation API

Production-grade RAG system for merchant cash advance (MCA) lender recommendations. Given a natural-language query (industry, state, revenue, positions), the system returns grounded, cited lender recommendations with faithfulness scoring.

## Stack

| Layer | Technology |
|---|---|
| Vector store | ChromaDB (persistent, 19 lender collections) |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| LLM | Groq `llama-3.3-70b-versatile` (temperature=0) |
| Faithfulness | `microsoft/deberta-v3-base-mnli` (NLI entailment) |
| API | FastAPI on port 8000 |

## How It Works

1. **Query understanding** — regex-first parser extracts intent, revenue, positions, state, industry in <1ms. LLM fallback only for ambiguous queries (~5% of traffic).
2. **Deterministic pre-filter** — 19 lenders are filtered against user criteria (revenue, position, state, industry) before any Chroma query runs.
3. **Semantic retrieval** — surviving lender collections are queried; results merged and capped at 1 chunk per lender for diversity.
4. **Grounded generation** — structured source headers with eligibility facts are injected into the prompt alongside explicit POSITION LOGIC and REVENUE LOGIC instructions.
5. **Faithfulness scoring** — every answer is scored via NLI entailment against cited sources (target ≥ 0.85, typical = 1.0).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Set environment variables:
```
GROQ_API_KEY=<your_key>
RAG_BACKEND=groq
```

## Build Embeddings

Run once (or after adding/updating guidelines):

```bash
python pre_processing/agent.py --dir guidelines/ --chroma chroma_db/
```

## Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Pharmacy in Iowa making 100k/month with 3 positions. Who can fund?", "collection": null, "tier": "balanced"}'
```

**Tiers:** `minimal` (~1s) · `balanced` (~1.6s, default) · `full` (~2–3s, with reranking)

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /query` | Submit a lender recommendation query |
| `GET /health` | Health check |
| `GET /collections` | List all lender collections |
| `GET /metrics` | Aggregated latency and faithfulness stats |

## Response Fields

```json
{
  "answer":              "For your pharmacy in Iowa...",
  "used_sources":        4,
  "collection":          "multi",
  "run_id":              "a729e983",
  "understand_ms":       0.1,
  "understand_used_llm": false,
  "retrieval_ms":        154.0,
  "llm_ms":              1251.0,
  "total_ms":            1618.0,
  "faithfulness":        1.0,
  "faithfulness_ms":     212.0
}
```

## Repository Layout

```
guidelines/          # 19 lender .txt files (source of truth)
pre_processing/
  agent.py           # ingestion: parse → chunk → embed → write to Chroma
unified_retrieval/
  backends/          # LLM backend abstraction (Groq / Ollama / Bedrock)
  config.py          # RAGConfig: tier definitions and env overrides
  faithfulness.py    # NLI faithfulness scorer
  monitoring.py      # PipelineMetrics dataclass + pluggable strategies
  query_improved.py  # semantic_query: MMR, cross-encoder rerank, neighbor expansion
  rag_qa.py          # main pipeline: understand → prefilter → retrieve → generate → score
api/
  app.py             # FastAPI application
chroma_db/           # ChromaDB persistent store (gitignored)
logs/
  rag_metrics.jsonl  # append-only metrics log
docs/
  SYSTEM_ARCHITECTURE.md  # full technical reference
```

For full implementation details see [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).
