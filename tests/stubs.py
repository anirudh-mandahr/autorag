"""Deterministic stand-ins for LLM and embedding clients (no network)."""

from __future__ import annotations

import hashlib
import json

from autorag.config import PipelineConfig
from autorag.usage import CallMetrics, UsageTracker

# Deterministic 8-dim unit vectors for known texts (cosine retrieval).
_EMBED_VECTORS: dict[str, list[float]] = {
    "alpha": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "beta": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "vault": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def _text_bucket(text: str) -> str:
    lowered = text.lower()
    if "alpha" in lowered or "orbit" in lowered:
        return "alpha"
    if "beta" in lowered or "cargo" in lowered or "harbor" in lowered:
        return "beta"
    if "vault" in lowered or "access code" in lowered:
        return "vault"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:8]


class StubEmbeddings:
    """Deterministic embeddings — no sentence-transformers load."""

    dimension = 8

    def __init__(self, usage: UsageTracker | None = None) -> None:
        self.usage = usage or UsageTracker()

    def _vector_for(self, text: str) -> list[float]:
        bucket = _text_bucket(text)
        if bucket in _EMBED_VECTORS:
            return list(_EMBED_VECTORS[bucket])
        raw = hashlib.sha256(text.encode("utf-8")).digest()
        return [((raw[i] / 255.0) * 2 - 1) for i in range(8)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            self.usage.record(
                CallMetrics(
                    latency_ms=0.1,
                    input_tokens=max(1, len(text.split())),
                    operation="embed_documents",
                )
            )
        return [self._vector_for(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.usage.record(
            CallMetrics(
                latency_ms=0.1,
                input_tokens=max(1, len(text.split())),
                operation="embed_query",
            )
        )
        return self._vector_for(text)


class StubLLM:
    """Deterministic LLM — no OpenRouter calls."""

    model_id = "stub-llm"
    provider = "stub"

    def __init__(self, usage: UsageTracker | None = None) -> None:
        self.usage = usage or UsageTracker()
        self.chat_calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
        operation: str = "chat",
        max_tokens: int | None = None,
    ) -> str:
        self.chat_calls.append(messages)
        self.usage.record(
            CallMetrics(
                latency_ms=1.0,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                operation=operation,
                model=self.model_id,
            )
        )
        user = messages[-1]["content"]
        system = messages[0].get("content", "").lower() if messages else ""
        if "correctness" in system or "expected answer" in user.lower():
            return json.dumps({"label": "correct"})
        if "supported" in system:
            return json.dumps({"supported": True})
        if "PipelineConfig" in user:
            return json.dumps(PipelineConfig(chunk_size=500).model_dump())
        return json.dumps({"supported": True})

    def grounded_answer(self, *, prompt: str) -> tuple[str, list[str]]:
        self.usage.record(
            CallMetrics(
                latency_ms=1.0,
                input_tokens=20,
                output_tokens=10,
                cost_usd=0.002,
                operation="answering",
                model=self.model_id,
            )
        )
        if "orbit" in prompt.lower() or "alpha" in prompt.lower():
            return (
                "Alpha station completes one orbit every 18 Earth days.",
                ["corpus-chunk-0"],
            )
        return ("Unknown.", ["corpus-chunk-0"])
