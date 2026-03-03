"""
RAG pipeline monitoring: metrics, timing, structured logging.
Modular design for cost/performance measurement and scaling.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# Metrics dataclass - single source of truth for pipeline measurements
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetrics:
    """Structured metrics for a single RAG query execution."""

    # Identity
    query: str = ""
    collection: str = ""
    run_id: str = ""

    # Retrieval stage
    retrieval_candidates: int = 0
    retrieval_after_mmr: int = 0
    retrieval_after_rerank: int = 0
    retrieval_after_expand: int = 0
    retrieval_after_filter: int = 0
    retrieval_ms: float = 0.0

    # LLM stage
    prompt_tokens_approx: int = 0
    completion_tokens_approx: int = 0
    llm_ms: float = 0.0
    llm_retries: int = 0

    # Output
    sources_used: int = 0
    answer_length: int = 0

    # Faithfulness (NLI entailment: answer grounded in sources)
    faithfulness: Optional[float] = None
    faithfulness_ms: float = 0.0
    unsupported_sentences: Optional[list] = None

    # Cost proxies (for scaling decisions)
    embedding_calls: int = 0
    cross_encoder_calls: int = 0

    # Errors
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self, indent: int = 0) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Context manager for timing
# ---------------------------------------------------------------------------


@contextmanager
def timed(metrics: PipelineMetrics, attr: str) -> Iterator[None]:
    """Record elapsed ms into metrics.{attr}."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(metrics, attr, getattr(metrics, attr, 0) + elapsed_ms)


# ---------------------------------------------------------------------------
# Logger setup - configurable, file + console
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log dir relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = "rag_pipeline.log",
    console: bool = True,
) -> logging.Logger:
    """Configure pipeline logger. Returns root RAG logger."""
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rag")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        root.addHandler(h)

    if log_file:
        fp = log_dir / log_file
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rag.{name}")


# ---------------------------------------------------------------------------
# Modular measurement strategies - pluggable callbacks
# ---------------------------------------------------------------------------


class MeasurementStrategy:
    """Base for pluggable measurement strategies."""

    def on_retrieval_start(self, query: str, collection: str) -> None:
        pass

    def on_retrieval_end(self, metrics: PipelineMetrics) -> None:
        pass

    def on_llm_start(self, metrics: PipelineMetrics) -> None:
        pass

    def on_llm_end(self, metrics: PipelineMetrics) -> None:
        pass

    def on_pipeline_end(self, metrics: PipelineMetrics) -> None:
        pass


class LoggingStrategy(MeasurementStrategy):
    """Log key metrics at each stage."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or get_logger("metrics")

    def on_retrieval_end(self, metrics: PipelineMetrics) -> None:
        self.log.info(
            "retrieval | collection=%s candidates=%d after_filter=%d ms=%.0f",
            metrics.collection,
            metrics.retrieval_candidates,
            metrics.retrieval_after_filter,
            metrics.retrieval_ms,
        )

    def on_llm_end(self, metrics: PipelineMetrics) -> None:
        self.log.info(
            "llm | ms=%.0f tokens_approx=%d+%d retries=%d",
            metrics.llm_ms,
            metrics.prompt_tokens_approx,
            metrics.completion_tokens_approx,
            metrics.llm_retries,
        )

    def on_pipeline_end(self, metrics: PipelineMetrics) -> None:
        self.log.info(
            "pipeline | sources=%d retrieval_ms=%.0f llm_ms=%.0f",
            metrics.sources_used,
            metrics.retrieval_ms,
            metrics.llm_ms,
        )


class JsonFileStrategy(MeasurementStrategy):
    """Append metrics as JSON lines to a file for downstream analysis."""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or (DEFAULT_LOG_DIR / "rag_metrics.jsonl")

    def on_pipeline_end(self, metrics: PipelineMetrics) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(metrics.to_json() + "\n")


class CostAwareStrategy(MeasurementStrategy):
    """Track cost proxies for scaling decisions."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or get_logger("cost")

    def on_pipeline_end(self, metrics: PipelineMetrics) -> None:
        # Approximate cost: embedding calls + cross_encoder + LLM tokens
        embed_cost = metrics.embedding_calls * 0.0001  # proxy
        ce_cost = metrics.cross_encoder_calls * 0.00005
        llm_tokens = metrics.prompt_tokens_approx + metrics.completion_tokens_approx
        llm_cost = llm_tokens * 0.000002  # proxy for local Ollama
        total = embed_cost + ce_cost + llm_cost
        self.log.info(
            "cost_proxy | embed_calls=%d ce_calls=%d llm_tokens=%d total_proxy=%.6f",
            metrics.embedding_calls,
            metrics.cross_encoder_calls,
            llm_tokens,
            total,
        )


# ---------------------------------------------------------------------------
# Global metrics collector - holds current run metrics
# ---------------------------------------------------------------------------

_STRATEGIES: list[MeasurementStrategy] = []


def register_strategy(s: MeasurementStrategy) -> None:
    _STRATEGIES.append(s)


def clear_strategies() -> None:
    _STRATEGIES.clear()


def notify(event: str, metrics: PipelineMetrics) -> None:
    """Emit event to all registered strategies."""
    for s in _STRATEGIES:
        m = getattr(s, event, None)
        if callable(m):
            m(metrics)
