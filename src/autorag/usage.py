"""Token, cost, and latency tracking for model calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CallMetrics:
    """Metrics for a single model call."""

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    operation: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageTracker:
    """Accumulates metrics across multiple model calls."""

    calls: list[CallMetrics] = field(default_factory=list)

    def record(self, metrics: CallMetrics) -> None:
        self.calls.append(metrics)

    @property
    def total_latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    def summary(self) -> dict[str, float | int]:
        return {
            "latency_ms": round(self.total_latency_ms, 2),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "calls": len(self.calls),
        }
