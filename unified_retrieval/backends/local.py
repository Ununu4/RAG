"""Local Ollama backend (free, no AWS)."""
from __future__ import annotations
from typing import Dict, List

import requests


class OllamaBackend:
    """Ollama LLM backend."""

    def __init__(self, url: str = None, model: str = None):
        self.url = url or "http://localhost:11434/api/chat"
        self.model = model or "nous-hermes2:latest"

    @property
    def name(self) -> str:
        return "ollama"

    def invoke(self, messages: List[Dict], temperature: float = 0.1, num_ctx: int = 12288, num_predict: int = 512, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
            "stream": False,
            "format": "json",
        }
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]
