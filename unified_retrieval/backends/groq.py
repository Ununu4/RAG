"""Groq backend (fast, free tier). Uses OpenAI-compatible API."""
from __future__ import annotations
import os
from typing import Dict, List

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqBackend:
    """Groq LLM backend."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required for Groq backend")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    @property
    def name(self) -> str:
        return "groq"

    def invoke(self, messages: List[Dict], temperature: float = 0.1, num_ctx: int = None, num_predict: int = 512, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
