# RAG Pipeline Internal Dynamics

## Execution Flow (Why We Got That Output)

For the query *"What is the minimum FICO score for Alternative Funding Group?"*:

### 1. Collection Selection
- **Input:** `--collection lender-alternative-funding-group` (explicit) or auto-detect
- **Auto-detect:** Tokenizes query, removes stop words (`funding`, `group`, etc.), matches against `lender-*` collection slugs. "alternative" overlaps → `lender-alternative-funding-group`
- **Output:** `chosen_collection = lender-alternative-funding-group`

### 2. Retrieval (query_improved.semantic_query)
| Stage | What Happens | Count |
|-------|--------------|-------|
| **Chroma query** | Embed query, fetch candidates | `candidate_k = max(30, 20)` |
| **MMR** | Maximal Marginal Relevance (λ=0.35) for diversity | `n_results * 2` or 10 |
| **Rerank** | Cross-encoder `ms-marco-MiniLM-L-6-v2` scores (query, doc) pairs | Top `n_results` |
| **Neighbor expand** | Follow `prev_id`/`next_id` links | +adjacent chunks |

### 3. Filtering
- **By lender:** Keep only docs where `lender_name` slug matches `alternative-funding-group`
- **Cross-lender:** Drop docs that mention other lender names (from content)

### 4. Prompt Construction
- **Sources block:** `[S1] id=... | lender=... | section=...` + doc preview (max 2000 chars each)
- **Schema:** `{"answer": "...", "used_sources": 0}`
- **System:** "Answer ONLY from provided sources. No citations in answer text."

### 5. LLM (Ollama)
- **Model:** `nous-hermes2:latest`
- **Options:** `temperature=0.1`, `num_ctx=12288`, `num_predict=512`, `format=json`
- **Output:** `{"answer": "The minimum FICO score for Alternative Funding Group is 550.", "used_sources": 1}`

---

## Why Sources Used = 1
The LLM returned `used_sources: 1` because the answer was concise and the model attributed it to a single source. The retrieved chunk `eligibility_criteria: ... FICO Score: Minimum 550 (Experian)` was clearly the primary evidence.

---

## Modular Measurement Strategies

| Strategy | Purpose | Output |
|----------|---------|--------|
| `LoggingStrategy` | Per-stage logs | `rag_pipeline.log` |
| `JsonFileStrategy` | JSONL metrics for dashboards | `rag_metrics.jsonl` |
| `CostAwareStrategy` | Cost proxies (embed/CE/LLM) | Log |

### Cost/Performance Knobs

| Knob | Default | Effect |
|------|---------|--------|
| `--n` | 6 | Fewer → faster retrieval, less context |
| `--expand` | 1 | 0 → no neighbor expansion, faster |
| `--rerank` | off | On → better precision, slower (CE inference) |
| `--doc-chars` | 2000 | Lower → smaller prompt, cheaper LLM |
| `--num-ctx` | 12288 | Lower → less memory, smaller context |
| `--num-predict` | 512 | Lower → shorter answers |

### Scaling Considerations
- **Embedding calls:** 1 (Chroma) + 1 (query) + 1 (docs) for MMR
- **Cross-encoder:** N calls when rerank enabled (N = docs after MMR)
- **LLM:** Single request; retry with trimmed prompt on 500
