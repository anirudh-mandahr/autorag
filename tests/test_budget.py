"""Budget reservation math."""

from __future__ import annotations

import pytest

from autorag.budget import (
    CostBreakdown,
    SpendLedger,
    estimate_eval_cost_usd,
    estimate_researcher_cost_usd,
    token_cost_usd,
)


def test_token_cost_matches_published_rates() -> None:
    assert token_cost_usd("meta-llama/llama-3.3-70b-instruct", 1_000_000, 0) == 0.10
    assert token_cost_usd("meta-llama/llama-3.3-70b-instruct", 0, 1_000_000) == 0.10


def test_estimates_are_positive_and_scale_with_queries() -> None:
    model = "meta-llama/llama-3.3-70b-instruct"
    one = estimate_eval_cost_usd(model, 1)
    ten = estimate_eval_cost_usd(model, 10)
    assert one > 0
    assert ten == pytest.approx(10 * one)
    assert estimate_researcher_cost_usd(model) > 0


def test_ledger_can_reserve_and_tracks_roles() -> None:
    ledger = SpendLedger()
    assert ledger.can_reserve(0.10, 0.04, holdback=0.03)
    ledger.add(CostBreakdown(researcher_usd=0.04, answering_usd=0.02, judging_usd=0.01))
    assert ledger.researcher_usd == 0.04
    assert ledger.answering_usd == 0.02
    assert ledger.judging_usd == 0.01
    assert ledger.total_usd == pytest.approx(0.07)
    assert not ledger.can_reserve(0.10, 0.04, holdback=0.0)
    assert ledger.can_reserve(0.10, 0.03, holdback=0.0)
