# modular_local_RAG

Local, privacy-preserving retrieval-augmented generation for lender policy FAQs. The repo has two modules:

- `pre_processing/agent.py`: ingests lender `.txt` files, chunks them, embeds with `multi-qa-MiniLM-L6-cos-v1`, and writes to Chroma collections named `lender-<slug>`.
- `unified_retrieval/`: serves answers from those embeddings.
  - `query_improved.py`: wraps Chroma semantic search with MMR, optional cross-encoder re-rank, and neighbor expansion.
  - `rag_qa.py`: retrieves lender-specific evidence, filters out cross-lender noise, and asks the LLM (Ollama/Groq/Bedrock) to return a structured answer with citations [S1], [S2].

## Requirements

- Python 3.11+
- Ollama running locally with `nous-hermes2:latest` pulled
- Local Chroma DB directory (defaults to `chroma_db/` in project root)
- Lender guidelines directory (defaults to `guidelines/` in project root)
- Python deps: `pip install -r requirements.txt`

### Optional/OS notes

- Windows users may need the Visual C++ build tools for `torch`.
- If Chroma requires an alternative SQLite backend on your machine, add an appropriate package (e.g., `pysqlite3-binary`) to `requirements.txt`.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate   # on Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
ollama pull nous-hermes2:latest
```

Ensure your data directories match the defaults or pass overrides via CLI flags.

## Usage

### 1) Build embeddings (pre-processing)

```bash
python pre_processing/agent.py --dir C:\path\to\lender_txts --chroma C:\path\to\Chromaa
```

What it does:
- Parses lender metadata headers from each `.txt`.
- Splits by headings, sentence-chunks text (size 180, overlap 40), tags chunks, extracts simple numeric hints, deduplicates near-duplicates.
- Stores chunks in Chroma with neighbor links (`prev_id`/`next_id`) for contextual expansion.

### 2) Query + answer (unified retrieval)

```bash
python unified_retrieval/rag_qa.py "Your lender-specific question" \
  --collection lender-alternative-funding-group \
  --chroma C:\path\to\Chromaa \
  --n 6 --expand 1 --rerank
```

What it does:
- Auto-detects the most likely `lender-*` collection if `--collection` is omitted.
- Retrieves evidence with MMR + optional cross-encoder rerank and neighbor expansion (via `query_improved.py`).
- Filters to the intended lender and drops chunks that mention other lenders.
- Prompts the LLM to return JSON with answer text and used_sources. Citations [S1], [S2] allowed.

Outputs: source count, latency metrics, and the answer text.

## Current development pipeline

1. Prepare lender `.txt` files with metadata headers (`LENDER_ID`, `LENDER_NAME`, `SOURCE_FILE`) followed by a dashed line and body content.
2. Run `agent.py` to (re)build Chroma collections under `lender-<slug>` names.
3. Serve queries locally by running `rag_qa.py` against the Chroma path and local Ollama.
4. Iterate on prompt/retrieval knobs (`--n`, `--expand`, `--rerank`, `--tier`) as needed. See `docs/RELIABILITY.md`.

## Repository layout

- `pre_processing/agent.py` — ingestion and embedding builder
- `unified_retrieval/query_improved.py` — retrieval helper (MMR/rerank/neighbor expansion)
- `unified_retrieval/rag_qa.py` — RAG QA pipeline against local Ollama
- `requirements.txt` — pinned dependencies
