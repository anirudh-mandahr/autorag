"""Thin OpenRouter client wrapper — sole entry point for LLM calls."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from autorag.config import PipelineConfig, Settings
from autorag.usage import CallMetrics, UsageTracker

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# USD per 1M tokens (OpenRouter list prices, approximate).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.10),
}


class LLMClient:
    """OpenRouter client with token, cost, and latency tracking."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        usage: UsageTracker | None = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self.usage = usage or UsageTracker()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Send a chat completion request and return assistant text."""
        if not self._settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        latency_ms = (time.perf_counter() - started) * 1000

        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        cost_usd = self._estimate_cost(input_tokens, output_tokens)

        self.usage.record(
            CallMetrics(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                model=self._settings.llm_model,
                operation="chat",
            )
        )

        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")
        return str(choices[0]["message"]["content"])

    def grounded_answer(
        self,
        *,
        prompt: str,
    ) -> tuple[str, list[str]]:
        """Ask the LLM for a grounded JSON answer with cited source ids."""
        raw = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return self._parse_grounded_response(raw)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = _MODEL_PRICING.get(
            self._settings.llm_model,
            (0.10, 0.10),
        )
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    @staticmethod
    def _parse_grounded_response(raw: str) -> tuple[str, list[str]]:
        text = raw.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                raise ValueError(f"LLM did not return JSON: {raw!r}") from None
            data = json.loads(match.group(0))

        answer = str(data.get("answer", "")).strip()
        source_ids = data.get("source_ids", [])
        if not isinstance(source_ids, list):
            source_ids = []
        return answer, [str(source_id) for source_id in source_ids]
