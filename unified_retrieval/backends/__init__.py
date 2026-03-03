"""LLM backends: local (Ollama) or AWS (Bedrock)."""
from .base import LLMBackend, get_backend

__all__ = ["LLMBackend", "get_backend"]
