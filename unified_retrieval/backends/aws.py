"""AWS Bedrock backend (free tier eligible)."""
from __future__ import annotations
import json
import os
from typing import Dict, List

try:
    import boto3
    from botocore.config import Config
    _BOTO_AVAILABLE = True
except ImportError:
    _BOTO_AVAILABLE = False

BEDROCK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION = "us-east-1"


class BedrockBackend:
    """Bedrock LLM backend."""

    def __init__(self, model_id: str = None, region: str = None):
        if not _BOTO_AVAILABLE:
            raise ImportError("boto3 required for AWS backend: pip install boto3")
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", BEDROCK_MODEL)
        self.region = region or os.getenv("AWS_REGION", BEDROCK_REGION)
        self._client = None

    @property
    def _bedrock(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(retries={"max_attempts": 3}),
            )
        return self._client

    @property
    def name(self) -> str:
        return "bedrock"

    def invoke(self, messages: List[Dict], temperature: float = 0.1, num_ctx: int = None, num_predict: int = 512, **kwargs) -> str:
        system = ""
        claude_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            else:
                claude_messages.append({"role": role, "content": content})

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": num_predict,
            "temperature": temperature,
            "system": system,
            "messages": claude_messages,
        }
        response = self._bedrock.invoke_model(modelId=self.model_id, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
