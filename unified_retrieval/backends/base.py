"""LLM backend abstraction for local vs AWS."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Dict, List


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def invoke(self, messages: List[Dict], **kwargs) -> str:
        """Invoke LLM and return response text."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier for metrics."""
        pass


def get_backend() -> LLMBackend:
    """Return backend from env: RAG_BACKEND=aws | groq | local (default)."""
    backend = (os.getenv("RAG_BACKEND") or "local").lower()
    if backend == "aws":
        from .aws import BedrockBackend
        return BedrockBackend()
    if backend == "groq":
        from .groq import GroqBackend
        return GroqBackend()
    from .local import OllamaBackend
    return OllamaBackend()
