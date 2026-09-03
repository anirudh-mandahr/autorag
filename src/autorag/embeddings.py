"""Sentence-transformers embedding wrapper with usage tracking."""

from __future__ import annotations

import time

from sentence_transformers import SentenceTransformer

from autorag.config import PipelineConfig, Settings
from autorag.usage import CallMetrics, UsageTracker


class EmbeddingClient:
    """HF sentence-transformers embeddings with token/cost/latency tracking."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        usage: UsageTracker | None = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self.usage = usage or UsageTracker()
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        model = self._get_model()
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension() or 0)
        return int(model.get_sentence_embedding_dimension() or 0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        started = time.perf_counter()
        vectors = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        token_estimate = sum(max(1, len(text.split())) for text in texts)

        self.usage.record(
            CallMetrics(
                latency_ms=latency_ms,
                input_tokens=token_estimate,
                output_tokens=0,
                cost_usd=0.0,
                model=self._config.embedding_model,
                operation="embed_documents",
            )
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        vector = self._get_model().encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        token_estimate = max(1, len(text.split()))

        self.usage.record(
            CallMetrics(
                latency_ms=latency_ms,
                input_tokens=token_estimate,
                output_tokens=0,
                cost_usd=0.0,
                model=self._config.embedding_model,
                operation="embed_query",
            )
        )
        return vector.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._config.embedding_model)
        return self._model
