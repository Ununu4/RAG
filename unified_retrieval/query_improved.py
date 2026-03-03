"""Simplified, high-signal Chroma query utility (MMR + optional re-rank + neighbor expansion)."""
from __future__ import annotations
import argparse
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder

# Paths / models
DEFAULT_CHROMA = r"C:\Users\ottog\desktop\Chromaa"
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # matches indexing
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_collection(path: str, name: str):
    client = PersistentClient(path=path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name, embedding_function=ef)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def _mmr(query_emb: np.ndarray, doc_embs: List[np.ndarray], k: int, lambda_param: float = 0.35) -> List[int]:
    selected: List[int] = []
    candidates = list(range(len(doc_embs)))
    if not candidates:
        return selected
    # First: pick best by relevance
    scores = [ _cos_sim(query_emb, d) for d in doc_embs ]
    first = int(np.argmax(scores))
    selected.append(first)
    candidates.remove(first)
    # Iteratively add diverse high-relevance items
    while len(selected) < min(k, len(doc_embs)) and candidates:
        best_idx, best_score = candidates[0], -1e9
        for i in candidates:
            relevance = _cos_sim(query_emb, doc_embs[i])
            redundancy = max(_cos_sim(doc_embs[i], doc_embs[j]) for j in selected) if selected else 0.0
            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr > best_score:
                best_idx, best_score = i, mmr
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


def semantic_query(
    query_text: str,
    collection_name: str,
    chroma_path: str = DEFAULT_CHROMA,
    n_results: int = 5,
    mmr: bool = True,
    rerank: bool = True,
    expand_neighbors: int = 0,
    metrics_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Semantic query with optional MMR, cross-encoder rerank, and neighbor expansion."""
    t0 = time.perf_counter()
    embed_calls = 0
    ce_calls = 0

    col = _get_collection(chroma_path, collection_name)

    # Fetch a larger candidate pool
    candidate_k = max(n_results * 5, 20)
    base = col.query(query_texts=[query_text], n_results=candidate_k)
    embed_calls += 1  # Chroma uses embedding fn for query

    ids = base["ids"][0]
    docs = base["documents"][0]
    metas = base["metadatas"][0]
    n_after_base = len(ids)

    # MMR diversity on SentenceTransformer embeddings
    if mmr and ids:
        st = SentenceTransformer(EMBED_MODEL)
        q_emb = st.encode(query_text)
        d_embs = st.encode(docs)
        embed_calls += 1 + 1  # query + docs
        pick = _mmr(q_emb, list(d_embs), k=min(len(ids), max(n_results * 2, 10)))
        ids, docs, metas = [ids[i] for i in pick], [docs[i] for i in pick], [metas[i] for i in pick]
    n_after_mmr = len(ids)

    # Cross-encoder rerank for precision
    if rerank and ids:
        try:
            ce = CrossEncoder(RERANK_MODEL)
            scores = ce.predict([[query_text, d] for d in docs])
            ce_calls += len(docs)
            order = list(np.argsort(scores)[::-1])[:n_results]
            ids, docs, metas = [ids[i] for i in order], [docs[i] for i in order], [metas[i] for i in order]
        except Exception as e:
            print(f"Warning: cross-encoder rerank failed ({e}); using pre-rerank order")
            ids, docs, metas = ids[:n_results], docs[:n_results], metas[:n_results]
    else:
        ids, docs, metas = ids[:n_results], docs[:n_results], metas[:n_results]
    n_after_rerank = len(ids)

    # Optional neighbor expansion using prev_id/next_id hints stored at ingest
    if expand_neighbors > 0 and ids:
        neighbor_ids: List[str] = []
        def _flatten_field(x):
            if not x:
                return []
            return x[0] if isinstance(x[0], list) else x
        def add_neighbors(mid: Dict, depth: int):
            left, right = mid.get("prev_id"), mid.get("next_id")
            ldepth = rdepth = depth
            cur_left, cur_right = left, right
            while ldepth > 0 and cur_left:
                neighbor_ids.append(cur_left)
                fetched = col.get(ids=[cur_left])
                next_meta = _flatten_field(fetched.get("metadatas", []))
                cur_left = next_meta[0].get("prev_id") if next_meta else None
                ldepth -= 1
            while rdepth > 0 and cur_right:
                neighbor_ids.append(cur_right)
                fetched = col.get(ids=[cur_right])
                next_meta = _flatten_field(fetched.get("metadatas", []))
                cur_right = next_meta[0].get("next_id") if next_meta else None
                rdepth -= 1

        for m in metas:
            add_neighbors(m, expand_neighbors)

        # Fetch unique neighbors not already included (preserve order)
        seen = set()
        filtered: List[str] = []
        for nid in neighbor_ids:
            if nid and (nid not in seen) and (nid not in ids):
                seen.add(nid)
                filtered.append(nid)
        if filtered:
            got = col.get(ids=filtered)
            ids.extend(_flatten_field(got.get("ids", [])))
            docs.extend(_flatten_field(got.get("documents", [])))
            metas.extend(_flatten_field(got.get("metadatas", [])))

    n_after_expand = len(ids)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if metrics_callback:
        metrics_callback({
            "retrieval_candidates": n_after_base,
            "retrieval_after_mmr": n_after_mmr,
            "retrieval_after_rerank": n_after_rerank,
            "retrieval_after_expand": n_after_expand,
            "retrieval_ms": retrieval_ms,
            "embedding_calls": embed_calls,
            "cross_encoder_calls": ce_calls,
        })

    return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [[0.0] * len(ids)]}


def print_results(results: Dict, query_text: str, show_full: bool = False):
    print("=" * 60)
    print(f"Query: '{query_text}'")
    print(f"Found {len(results['ids'][0])} results:")
    print("=" * 60)
    for i, (rid, doc, meta) in enumerate(zip(results["ids"][0], results["documents"][0], results["metadatas"][0]), 1):
        lender = meta.get("lender_name", "?")
        section = meta.get("section", "")
        tags = meta.get("tags", "")
        hints = []
        for k in ("min_fico", "min_revenue", "min_tib_months"):
            if meta.get(k):
                hints.append(f"{k}={meta[k]}")
        hint_str = ", ".join(hints)
        print("-" * 60)
        print(f"{i}. [{lender}] ({section})  id={rid}")
        if tags:
            print(f"   tags: {tags}")
        if hint_str:
            print(f"   hints: {hint_str}")
        if show_full:
            print(f"   {doc}")
        else:
            preview = doc if len(doc) <= 300 else doc[:300] + "..."
            print(f"   {preview}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple semantic query with MMR/rerank")
    p.add_argument("query", help="Search query text")
    p.add_argument("--collection", default="lender-alternative-funding-group")
    p.add_argument("--chroma", default=DEFAULT_CHROMA)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--no-mmr", dest="mmr", action="store_false")
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--expand", type=int, default=0, help="Neighbor depth to expand (per side)")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()

    res = semantic_query(
        args.query,
        collection_name=args.collection,
        chroma_path=args.chroma,
        n_results=args.n,
        mmr=args.mmr,
        rerank=args.rerank,
        expand_neighbors=args.expand,
    )
    print_results(res, args.query, show_full=args.full)

