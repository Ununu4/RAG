# RAG Pipeline Reliability

Design choices for accuracy and reliability in the RAG demo.

## Retrieval

| Component | Purpose |
|-----------|---------|
| **MMR (Maximal Marginal Relevance)** | Balances relevance and diversity—avoids redundant chunks, improves coverage |
| **Lender filter** | Keeps only docs from the target lender (metadata match) |
| **Cross-lender filter** | Drops chunks that mention other lenders in content—reduces confusion |
| **Collection auto-detect** | Token overlap between query and collection slugs—picks lender when not specified |

## Generation

| Component | Purpose |
|-----------|---------|
| **Simple prompt** | Short system + user prompt—fewer model errors, faster inference |
| **Citations [S1], [S2]** | Explicit source IDs—traceable, verifiable answers |
| **Minimal JSON schema** | `{"answer": "...", "used_sources": N}`—easy to parse, fewer format errors |

## Parsing

| Component | Purpose |
|-----------|---------|
| **Regex fallback** | If `json.loads` fails, extract first `{...}` block—handles markdown-wrapped JSON |
| **Double-JSON unwrap** | If `answer` contains nested JSON string, parse and extract inner `answer`—handles model quirks |

## Metrics (Removed)

- **Coherence** removed—embedding similarity between answer bullets and docs was noisy and not actionable for demo.

## Metrics (Kept)

- **retrieval_ms, llm_ms, total_ms**—latency for tuning
- **used_sources**—model-reported citation count
- **run_id**—traceability

## Tiers

| Tier | n_results | expand | num_predict | Use case |
|------|-----------|--------|------------|----------|
| minimal | 3 | 0 | 384 | Fast demo |
| balanced | 6 | 1 | 512 | Default quality |
| full | 8 | 2 | 768 | Best quality, rerank on |
