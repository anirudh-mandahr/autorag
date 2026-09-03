"""Hard USD spend cap: conservative pre-call reservation and cost ledgers.

``MAX_USD_SPEND`` is a **hard per-session cap**. The research loop will not start a
researcher proposal or an evaluation unless the remaining budget covers a
conservative maximum-cost estimate derived from provider token limits. Actual
billed cost can still theoretically exceed an estimate if a provider ignores
``max_tokens``; in that case the loop records the overshoot and stops without
starting further paid work.

Embedding calls are treated as $0 in the current pricing table.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens (OpenRouter list prices, approximate).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.10),
}
DEFAULT_RATES: tuple[float, float] = (0.10, 0.10)

# Conservative caps also passed to the provider as max_tokens (output) and used
# to size the input side of the reservation.
RESEARCHER_MAX_INPUT_TOKENS = 8_192
RESEARCHER_MAX_OUTPUT_TOKENS = 1_024
ANSWER_MAX_INPUT_TOKENS = 4_096
ANSWER_MAX_OUTPUT_TOKENS = 512
JUDGE_MAX_INPUT_TOKENS = 4_096
JUDGE_MAX_OUTPUT_TOKENS = 256

_EPS = 1e-12


def token_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a token pair at the model's published rates."""
    input_rate, output_rate = MODEL_PRICING.get(model, DEFAULT_RATES)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def estimate_researcher_cost_usd(model: str) -> float:
    """Maximum reserved cost for one researcher proposal call."""
    return token_cost_usd(
        model,
        RESEARCHER_MAX_INPUT_TOKENS,
        RESEARCHER_MAX_OUTPUT_TOKENS,
    )


def estimate_eval_cost_usd(
    model: str,
    n_queries: int,
    *,
    n_judges: int = 2,
) -> float:
    """Maximum reserved cost for evaluating ``n_queries`` (answer + judges).

    Assumes every question is answered and judged ``n_judges`` times (groundedness
    and correctness). Refusals cost less; the reserve is intentionally conservative.
    """
    per_answer = token_cost_usd(model, ANSWER_MAX_INPUT_TOKENS, ANSWER_MAX_OUTPUT_TOKENS)
    per_judge = token_cost_usd(model, JUDGE_MAX_INPUT_TOKENS, JUDGE_MAX_OUTPUT_TOKENS)
    return max(0, n_queries) * (per_answer + n_judges * per_judge)


@dataclass
class CostBreakdown:
    """USD spent in one phase, split by role."""

    researcher_usd: float = 0.0
    answering_usd: float = 0.0
    judging_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.researcher_usd + self.answering_usd + self.judging_usd


@dataclass
class SpendLedger:
    """Accumulates session spend by role for cap enforcement."""

    researcher_usd: float = 0.0
    answering_usd: float = 0.0
    judging_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.researcher_usd + self.answering_usd + self.judging_usd

    def add(self, breakdown: CostBreakdown) -> None:
        self.researcher_usd += breakdown.researcher_usd
        self.answering_usd += breakdown.answering_usd
        self.judging_usd += breakdown.judging_usd

    def remaining(self, cap: float) -> float:
        return cap - self.total_usd

    def can_reserve(self, cap: float, estimate: float, *, holdback: float = 0.0) -> bool:
        """True iff remaining budget covers ``estimate`` while leaving ``holdback``."""
        return self.remaining(cap) + _EPS >= estimate + holdback
