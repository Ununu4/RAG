"""
NLI-based faithfulness scoring: measures whether the answer is entailed by the sources.
Uses textual entailment (premise=context, hypothesis=claim) for objective truth grounding.
Falls back to embedding similarity if NLI model cannot be loaded (e.g. offline, auth).
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

_NLI_MODEL = None
_NLI_TOKENIZER = None
_USE_EMBEDDING_FALLBACK = False
_EMBED_MODEL = None


def _get_nli_model():
    """Lazy-load NLI model (DeBERTa-v3-MNLI) for entailment checking."""
    global _NLI_MODEL, _NLI_TOKENIZER, _USE_EMBEDDING_FALLBACK
    if _NLI_MODEL is None and not _USE_EMBEDDING_FALLBACK:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "microsoft/deberta-v3-base-mnli"
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
            _NLI_MODEL.eval()
        except Exception:
            _USE_EMBEDDING_FALLBACK = True
    if _USE_EMBEDDING_FALLBACK:
        return None, None
    return _NLI_MODEL, _NLI_TOKENIZER


def _get_embed_model():
    """Lazy-load sentence encoder for fallback similarity-based faithfulness."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    return _EMBED_MODEL


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering trivial ones."""
    if not (text or "").strip():
        return []
    # Simple split: period, exclamation, question mark, newline
    raw = re.split(r"[.!?]\s+|\n+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short, citations-only, or JSON-like
        if len(s) < 10:
            continue
        if re.match(r"^\[S\d+\]\s*$", s):
            continue
        if s.startswith("{") or s.startswith("}"):
            continue
        sentences.append(s)
    return sentences if sentences else [text.strip()] if text.strip() else []


def _extract_cited_sources(sentence: str) -> List[int]:
    """Extract source indices from [S1], [S2], etc. Returns 1-based indices (S1=1, S2=2)."""
    indices: List[int] = []
    for m in re.finditer(r"\[S(\d+)\]", sentence):
        idx = int(m.group(1))
        if idx > 0 and idx not in indices:
            indices.append(idx)
    return indices


def _truncate_for_nli(text: str, max_chars: int = 1800) -> str:
    """Truncate context to fit NLI model (~512 tokens, ~4 chars/token)."""
    if not text or len(text) <= max_chars:
        return (text or "").strip()
    return text[:max_chars].rsplit(" ", 1)[0] + "..." if " " in text[:max_chars] else text[:max_chars]


def _compute_faithfulness_nli(
    sentences: List[str],
    context: str,
    entailment_threshold: float,
    docs: Optional[List[str]] = None,
    max_context_chars: int = 1800,
) -> Tuple[int, List[str]]:
    """NLI-based: premise=context, hypothesis=claim. Citation-aware: use cited docs as context when available."""
    import torch
    model, tokenizer = _get_nli_model()
    if model is None or tokenizer is None:
        return -1, []  # signal fallback needed

    supported = 0
    unsupported: List[str] = []
    base_context = _truncate_for_nli(context, max_context_chars)

    for sent in sentences:
        if len(sent) < 5:
            supported += 1
            continue
        # Citation-aware: use cited source docs as context for better entailment
        ctx = base_context
        if docs and _extract_cited_sources(sent):
            cited = _extract_cited_sources(sent)
            parts = []
            for idx in cited:
                if 1 <= idx <= len(docs) and isinstance(docs[idx - 1], str):
                    parts.append(docs[idx - 1].strip())
            if parts:
                ctx = _truncate_for_nli(" ".join(parts), max_context_chars)
        inputs = tokenizer(
            ctx,
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        entailment_score = probs[0][0].item()
        if entailment_score >= entailment_threshold:
            supported += 1
        else:
            unsupported.append(sent)
    return supported, unsupported


def _compute_faithfulness_embedding(
    sentences: List[str],
    docs: List[str],
    similarity_threshold: float = 0.5,
) -> Tuple[int, List[str]]:
    """Fallback: max cosine similarity of sentence to any source chunk above threshold = supported."""
    import numpy as np
    model = _get_embed_model()
    doc_embs = model.encode(docs) if docs else np.array([])
    if doc_embs.size == 0:
        return 0, sentences
    supported = 0
    unsupported: List[str] = []
    for sent in sentences:
        if len(sent) < 5:
            supported += 1
            continue
        sent_emb = model.encode(sent)
        sims = np.dot(doc_embs, sent_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(sent_emb) + 1e-9
        )
        max_sim = float(np.max(sims))
        if max_sim >= similarity_threshold:
            supported += 1
        else:
            unsupported.append(sent)
    return supported, unsupported


def compute_faithfulness(
    answer_text: str,
    results: Dict,
    entailment_threshold: float = 0.45,
    max_context_chars: int = 2400,
) -> Tuple[float, List[str], float]:
    """
    Compute faithfulness score via NLI entailment (best for truth grounding).
    Falls back to embedding similarity if NLI model unavailable.

    faithfulness = (# supported sentences) / (# total sentences).

    Returns:
        (score, unsupported_sentences, elapsed_ms)
    """
    sentences = _split_sentences(answer_text)
    if not sentences:
        return 1.0, [], 0.0

    docs = results.get("documents", [[]])
    docs = docs[0] if docs else []
    context = " ".join(d for d in docs if isinstance(d, str) and d.strip())
    context = _truncate_for_nli(context, max_context_chars)
    if not context.strip():
        return 0.0, sentences, 0.0

    t0 = time.perf_counter()
    supported, unsupported = _compute_faithfulness_nli(
        sentences, context, entailment_threshold,
        docs=docs, max_context_chars=max_context_chars,
    )
    if supported < 0:
        supported, unsupported = _compute_faithfulness_embedding(
            sentences, docs, similarity_threshold=0.32
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    score = supported / len(sentences) if sentences else 1.0
    return round(score, 4), unsupported, round(elapsed_ms, 2)
