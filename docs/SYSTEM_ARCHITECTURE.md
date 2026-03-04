# RAG System Architecture — Complete Technical Reference

> **Purpose:** Atomic, granular documentation of the lender-recommendation RAG system.  
> Enables any LLM or engineer to understand every component, data flow, and decision without reading source code.

---

## 1. System Overview

**Domain:** Merchant cash advance (MCA) lender recommendations.  
**Input:** Natural-language query (industry, state, monthly revenue, existing positions).  
**Output:** Grounded, cited answer with 3–5 lender recommendations and faithfulness score.

**Pipeline (6 stages):**
1. **Query understanding** — regex-first parser extracts intent, revenue, positions, state, industry (<5ms). LLM fallback only for ambiguous queries (~5%).
2. **Deterministic pre-filter** — 19 lenders filtered by revenue, position, state, industry before any Chroma query.
3. **Semantic retrieval** — ChromaDB search across surviving collections; results merged with lender diversity cap.
4. **Source formatting** — structured headers (min rev, positions, restricted states) prepended to each chunk.
5. **LLM generation** — Groq `llama-3.3-70b-versatile` with explicit POSITION/REVENUE logic; JSON output.
6. **Faithfulness scoring** — NLI entailment (DeBERTa-v3-MNLI) per sentence against cited sources.

**Query modes:**
- **which_lender** + `collection=null` → multi-collection search; returns 3–5 recommendations.
- **Single-lender** → search one collection; returns detailed policy info.

---

## 2. Repository Layout

```
RAG/
├── guidelines/                     # 19 lender .txt files (source of truth)
├── pre_processing/
│   └── agent.py                    # Ingestion: parse → chunk → embed → Chroma
├── unified_retrieval/
│   ├── backends/
│   │   ├── base.py                 # LLMBackend ABC, get_backend()
│   │   ├── groq.py                 # Groq (llama-3.3-70b-versatile)
│   │   ├── local.py                # Ollama
│   │   └── aws.py                  # Bedrock
│   ├── config.py                   # RAGConfig, tier definitions
│   ├── faithfulness.py             # NLI faithfulness scorer
│   ├── monitoring.py               # PipelineMetrics, MeasurementStrategy
│   ├── query_improved.py           # semantic_query (MMR, rerank, neighbors)
│   └── rag_qa.py                   # Main pipeline
├── api/
│   └── app.py                      # FastAPI: /query, /health, /collections, /metrics
├── chroma_db/                      # ChromaDB persistent store (19 collections)
├── logs/
│   └── rag_metrics.jsonl           # Append-only metrics
└── docs/
    └── SYSTEM_ARCHITECTURE.md
```

---

## 3. Data Layer — Guidelines Files

**Location:** `guidelines/*.txt`  
**Count:** 19 lenders  
**Format:** Canonical schema

```
# LENDER: <slug>

## funding_guidelines
<bullet list: amounts, terms, products, commissions>

## eligibility_criteria
<bullet list: TIB, FICO, revenue, positions, states, industries>

## deal_structure
<MCA/loan structure>

## key_findings
<summary bullets>
```

**Canonical sections:** `funding_guidelines`, `eligibility_criteria`, `deal_structure`, `key_findings`.

**Legacy format supported:** `### LENDER_ID:`, `### LENDER_NAME:`, `### SOURCE_FILE:`, `-----`, then body.

---

## 4. Pre-Processing Pipeline

**File:** `pre_processing/agent.py`  
**Entry:** `process_directory(txt_dir, chroma_path)`

### 4.1 Constants

| Constant | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 320 | Tokens per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap tokens |
| `SMALL_SECTION_CHARS` | 600 | Keep whole if smaller |
| `INDUSTRY_LIST_CHARS` | 1600 | Split long comma lists |
| `INDUSTRY_BATCH_SIZE` | 20 | Items per batch |
| `EMBED_MODEL` | `multi-qa-MiniLM-L6-cos-v1` | Must match query-time |

### 4.2 Flow

1. **Ingest** — `FileFetcher` + `TextChef` reads `.txt` files.
2. **Parse** — `parse_lender_file()` extracts metadata (`lender_id`, `lender_name`, `source_file`) and body.
3. **Section split** — `split_sections()` detects `##` headings, bold headings, named sections.
4. **Subheading split** — `_split_by_subheadings()` splits on `- **SubHeading:**`.
5. **Chunk routing:**
   - `< 600 chars` → keep whole
   - Long comma list (≥10 items, ≥1600 chars) → `_maybe_split_industry_list()` batches of 20
   - Else → `SentenceChunker(chunk_size=320, overlap=64)`
6. **Prefix** — Each chunk gets `section: subheading: ` prepended.
7. **Deduplication** — Hash of normalized text; skip duplicates.
8. **Metadata** — `section`, `section_index`, `canonical_section_index`, `chunk_index`, `tags`, `min_fico`, `min_revenue`, `min_tib_months`.
9. **Neighbor links** — `prev_id`, `next_id` within same section.
10. **Embed** — `EmbeddingsRefinery(embedding_model=EMBED_MODEL)`.
11. **Write** — `collection.add(ids, documents, embeddings, metadatas)` per lender.

**Collection naming:** `lender-<slug>` (e.g. `lender-bitty-advance`).  
**Chunk IDs:** `{collection_name}-{section_index}-{chunk_index}`.

---

## 5. Vector Store (ChromaDB)

**Path:** `chroma_db/` (or `RAG_CHROMA_PATH`)  
**Collections:** 19, one per lender  
**Client caching:** `_CHROMA_CLIENTS` dict in `query_improved.py` caches `PersistentClient` per path.

**Lender → collection mapping:** `lender-501-advance`, `lender-advantage-capital-funding`, `lender-alternative-funding-group`, `lender-apex-funding-source`, `lender-arsenal-funding`, `lender-aspire-funding-platform`, `lender-aurum-funding`, `lender-avanza`, `lender-backd`, `lender-bellwether`, `lender-bitty-advance`, `lender-biz-2-credit`, `lender-bizfund`, `lender-blade`, `lender-can-capital`, `lender-cashable`, `lender-cfg-merchant-solutions`, `lender-channel`, `lender-clearfund`.

---

## 6. Retrieval Layer

**File:** `unified_retrieval/query_improved.py`  
**Function:** `semantic_query(query_text, collection_name, chroma_path, n_results, mmr, rerank, expand_neighbors, metrics_callback)`

### 6.1 Models (lazy-loaded, cached)

| Model | Use |
|---|---|
| `multi-qa-MiniLM-L6-cos-v1` | Embeddings (Chroma + MMR) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking (optional) |

### 6.2 Pipeline

1. **Base fetch** — `candidate_k = max(n*5, 20)` if MMR else `n_results`; `col.query()`.
2. **MMR** (if enabled) — `lambda=0.35`; relevance vs redundancy; pick up to `min(len, max(n*2, 10))`.
3. **Rerank** (if enabled) — Cross-encoder scores `(query, doc)`; reorder and trim to `n_results`.
4. **Neighbor expansion** (if `expand_neighbors > 0`) — Walk `prev_id`/`next_id`; fetch neighbors by ID.
5. **Metrics callback** — Reports `retrieval_candidates`, `retrieval_after_mmr`, `retrieval_after_rerank`, `retrieval_after_expand`, `retrieval_ms`, `embedding_calls`, `cross_encoder_calls`.

---

## 7. RAG Pipeline — Core Logic

**File:** `unified_retrieval/rag_qa.py`  
**Function:** `answer_query(query, collection, chroma_path, ...)`

### 7.1 Stage 0: Query Understanding — `_understand_query(query, metrics)`

**Regex parser** `_parse_query_regex()` extracts:
- **Intent** — Phrase lists: `_WHICH_LENDER_PHRASES`, `_REQUIREMENTS_PHRASES`, `_ELIGIBILITY_PHRASES`, `_RESTRICTIONS_PHRASES`
- **Revenue** — `$80K`, `80k`, `$80,000`, `80000 per month`
- **Positions** — `2 positions`, `two positions`
- **State** — Full names (`_STATE_MAP`) or `in/from/based in [A-Z]{2}`
- **Industry** — 21 regex patterns (`_INDUSTRY_PATTERNS`)
- **Lender** — `_build_lender_lookup()` from `LENDER_ELIGIBILITY`

**LLM fallback:** `use_llm = intent == "other" and confidence < 2`. LLM fills gaps; regex values take precedence.

**Intent upgrade:** If `intent == "other"` but revenue/positions/state found and query has funding keywords → `which_lender`.

**Revenue normalization:** `_normalize_revenue()` — e.g. `90` + "90K" in query → `90000`.

**Metrics:** `understand_ms`, `understand_used_llm`.

### 7.2 Search Query Expansion

Appends to query: `{industry} industry eligibility`, `{state} state restrictions`, `revenue minimum monthly requirements`, `positions auto decline eligibility` when present in `intent_context`.

### 7.3 Stage 1a: Multi-Lender Retrieval

**Condition:** `intent == "which_lender"` and `collection is None`.

**Pre-filter** `_prefilter_collections(collections, user_criteria)`:
- Revenue: skip if `user_revenue < lender.min_revenue`
- Position: `next_position = positions + 1`; skip if `next_position > lender.max_position`
- State: skip if `user_state in lender.restricted_states`
- Industry: skip if any `prohibited_keyword in user_industry`

**Multi-collection search:** For each surviving collection, `semantic_query(n_per_collection=2, mmr=False, rerank=False)`. Merge, sort by distance, cap `max_per_lender=1`, up to `min(n_results*2, 10)` docs.

### 7.4 Stage 1b: Single-Lender Retrieval

- `_resolve_collection(short_name)` — e.g. `"bitty"` → `lender-bitty-advance`
- `_detect_collection_for_query()` — token overlap if no collection specified
- `semantic_query(mmr=True, rerank=use_rerank, expand_neighbors=expand_neighbors)`
- `_filter_results_by_lender()` — keep only chunks with matching `lender_name`
- `_filter_cross_lender_mentions()` — drop chunks mentioning other lenders in text

### 7.5 Stage 2: Source Formatting — `_format_sources(results, max_chars_per_doc)`

Per chunk:
```
[S{i}] {display_name} | Min Rev: ${min}/mo | Positions: up to {max}th pos | Restricted states: {states}
<chunk text, truncated at max_chars_per_doc>
```

Uses `LENDER_ELIGIBILITY` for headers; fallback `id=|lender=|section=` if not in registry.

### 7.6 Stage 2b: Background Distillation

For `which_lender`: `_distill_sources(results, max_bullets=25)` — first 25 bullet lines as compact "Key points" block.

### 7.7 Stage 3: Prompt Construction — `_build_messages()`

**System:** Trusted advisor; cite [S1], [S2]; exact numbers; no inference; JSON only.

**User (which_lender):** Context with criteria; POSITION LOGIC (next_pos, exclude rules); REVENUE LOGIC (exclude if min > user); distilled bullets; full sources; JSON schema.

**JSON schema:**
- which_lender: `{"intro": "...", "lenders": ["Lender [S#]: ..."], "used_sources": 0}`
- Other: `{"answer": "...", "used_sources": 0}`

### 7.8 Stage 4: LLM Invocation

`_invoke_llm(messages, num_ctx, num_predict)` → `get_backend().invoke()`.  
**Retry on 500:** Trimmed prompt (60% doc chars, `num_ctx=8192`).

### 7.9 Stage 5: JSON Parsing

1. `json.loads(raw)`
2. `re.search(r"\{[\s\S]*\}")` + parse
3. Fallback `{"answer": raw, "used_sources": 0}`

**Unwrap:** `intro` + `lenders[]` → joined; or `answer` string/list.

### 7.10 Stage 6: Faithfulness

`compute_faithfulness(answer_text, results)` — see Section 9.

### 7.11 Return

`{"json": obj, "answer_text": str, "metrics": PipelineMetrics}`

---

## 8. Lender Eligibility Registry

**Location:** `LENDER_ELIGIBILITY` dict in `rag_qa.py`  
**Schema per lender:**
```python
{
    "display_name": str,
    "min_revenue": int | None,      # dollars
    "max_position": int,             # 99 = no limit
    "restricted_states": List[str],  # e.g. ["CA", "NY"]
    "prohibited_keywords": List[str], # e.g. ["pharmacy", "cannabis"]
}
```

**Used for:** Pre-filtering and source headers. Must be kept in sync with guidelines.

---

## 9. Faithfulness Scoring

**File:** `unified_retrieval/faithfulness.py`  
**Function:** `compute_faithfulness(answer_text, results, entailment_threshold=0.45, max_context_chars=2400)`

**Model:** `microsoft/deberta-v3-base-mnli` (lazy-loaded).

**Algorithm:**
1. `_split_sentences()` — split on `[.!?\n]`; filter short, citation-only, JSON-like.
2. Per sentence: extract `[S#]` citations; if cited, use only cited docs as context; else full context.
3. Truncate context to `max_context_chars`.
4. NLI: premise=context, hypothesis=sentence; entailment score = softmax[0][0].
5. Supported if `score >= 0.45`.
6. `faithfulness = supported / total`.

**Fallback:** Embedding similarity (threshold 0.32) if NLI fails to load.

**Returns:** `(score, unsupported_sentences, elapsed_ms)`.

---

## 10. LLM Backend

**Selection:** `RAG_BACKEND` env: `groq` | `aws` | `local`.

**Groq** (`backends/groq.py`):
- URL: `https://api.groq.com/openai/v1/chat/completions`
- Model: `llama-3.3-70b-versatile` (or `GROQ_MODEL`)
- `temperature=0`, `response_format={"type": "json_object"}`
- Auth: `GROQ_API_KEY`

---

## 11. Configuration and Tiers

**File:** `unified_retrieval/config.py`

| Parameter | minimal | balanced | full |
|---|---|---|---|
| n_results | 3 | 6 | 8 |
| expand_neighbors | 0 | 1 | 2 |
| use_rerank | False | False | True |
| max_chars_per_doc | 1200 | 1500 | 2500 |
| num_ctx | 8192 | 8192 | 16384 |
| num_predict | 384 | 700 | 768 |

**Env overrides:** `RAG_N_RESULTS`, `RAG_EXPAND`, `RAG_RERANK`, `RAG_DOC_CHARS`, `RAG_NUM_CTX`, `RAG_NUM_PREDICT`, `RAG_TIER`, `RAG_FAITHFULNESS`.

---

## 12. Monitoring

**File:** `unified_retrieval/monitoring.py`

**PipelineMetrics fields:** `query`, `collection`, `run_id`, `understand_ms`, `understand_used_llm`, `retrieval_*`, `llm_ms`, `prompt_tokens_approx`, `completion_tokens_approx`, `sources_used`, `answer_length`, `faithfulness`, `faithfulness_ms`, `faithfulness_error`, `embedding_calls`, `cross_encoder_calls`, `error`.

**Strategies:** `LoggingStrategy`, `JsonFileStrategy`, `CostAwareStrategy`.  
**Events:** `on_retrieval_end`, `on_llm_start`, `on_llm_end`, `on_pipeline_end`.

---

## 13. FastAPI Application

**File:** `api/app.py`  
**Run:** `uvicorn api.app:app --host 0.0.0.0 --port 8000`

**Endpoints:**
- `POST /query` — Body: `{query, collection?, tier}`; returns answer + metrics.
- `GET /health` — `{"status":"ok"}`
- `GET /collections` — List lender collections
- `GET /metrics?limit=100` — Aggregated latency and faithfulness

**Middleware:** Sanitizes control chars in POST /query body.

**QueryResponse:** `answer`, `used_sources`, `collection`, `run_id`, `understand_ms`, `understand_used_llm`, `retrieval_ms`, `llm_ms`, `total_ms`, `faithfulness`, `faithfulness_ms`, `faithfulness_error`.

---

## 14. Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for Groq |
| `RAG_BACKEND` | `groq` \| `aws` \| `local` |
| `RAG_CHROMA_PATH` | ChromaDB path |
| `RAG_TIER` | `minimal` \| `balanced` \| `full` |
| `RAG_FAITHFULNESS` | `0`/`false` to disable |
| `PYTHONPATH` | Must include `unified_retrieval` |

---

## 15. Data Contracts

**intent_context:**
```python
{"industry", "lender", "intent", "revenue_monthly", "positions", "state"}
# intent: which_lender | eligibility | requirements | restrictions | comparison | other
```

**Chunk metadata:**
```python
{"lender_id", "lender_name", "source_file", "section", "section_index",
 "canonical_section_index", "chunk_index", "tags", "prev_id", "next_id",
 "min_fico", "min_revenue", "min_tib_months"}
```

**answer_query return:**
```python
{"json": dict, "answer_text": str, "metrics": PipelineMetrics}
```

---

## 16. End-to-End Flow (which_lender)

```
POST /query {"query": "Pharmacy in Iowa 100k/mo 3 positions. Who can fund?", "tier": "balanced"}
  → RAGConfig._apply_tier("balanced")
  → answer_query()
    → _understand_query() [regex: intent=which_lender, rev=100000, pos=3, state=IA, industry=pharmacy]
    → search_query = query + " pharmacy industry eligibility IA state restrictions revenue minimum monthly requirements positions auto decline eligibility"
    → _multi_collection_search(search_query, user_criteria)
      → _prefilter_collections() [e.g. BackD: pharmacy prohibited → skip; BizFund: max_pos=3, next=4 → skip]
      → semantic_query() per surviving collection (mmr=False)
      → merge, sort by distance, cap 1 per lender
    → _format_sources() with LENDER_ELIGIBILITY headers
    → _distill_sources() for Key points
    → _build_messages() with POSITION/REVENUE logic
    → _invoke_llm() [Groq llama-3.3-70b]
    → Parse JSON, unwrap intro + lenders
    → compute_faithfulness()
  → QueryResponse
```

---

*Document reflects current codebase state. Update when architecture changes.*
