"""
Cost/performance config for RAG pipeline.
Override via environment variables for scaling (e.g. RAG_N_RESULTS=4).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    """Pipeline config with env overrides."""

    n_results: int = 6
    expand_neighbors: int = 1
    use_rerank: bool = False
    max_chars_per_doc: int = 2000
    num_ctx: int = 12288
    num_predict: int = 512

    # Cost tiers: "minimal" | "balanced" | "full"
    tier: str = "balanced"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        c = cls()
        if os.getenv("RAG_N_RESULTS"):
            c.n_results = int(os.environ["RAG_N_RESULTS"])
        if os.getenv("RAG_EXPAND"):
            c.expand_neighbors = int(os.environ["RAG_EXPAND"])
        if os.getenv("RAG_RERANK", "").lower() in ("1", "true", "yes"):
            c.use_rerank = True
        if os.getenv("RAG_DOC_CHARS"):
            c.max_chars_per_doc = int(os.environ["RAG_DOC_CHARS"])
        if os.getenv("RAG_NUM_CTX"):
            c.num_ctx = int(os.environ["RAG_NUM_CTX"])
        if os.getenv("RAG_NUM_PREDICT"):
            c.num_predict = int(os.environ["RAG_NUM_PREDICT"])
        if os.getenv("RAG_TIER"):
            c.tier = os.environ["RAG_TIER"].lower()
            c._apply_tier()
        return c

    def _apply_tier(self) -> None:
        if self.tier == "minimal":
            self.n_results = 3
            self.expand_neighbors = 0
            self.use_rerank = False
            self.max_chars_per_doc = 1200
            self.num_ctx = 8192
            self.num_predict = 256
        elif self.tier == "full":
            self.n_results = 8
            self.expand_neighbors = 2
            self.use_rerank = True
            self.max_chars_per_doc = 2500
            self.num_ctx = 16384
            self.num_predict = 768
