"""Research loop budget caps and config dedup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorag.budget import CostBreakdown
from autorag.config import PipelineConfig, Settings, compute_objective
from autorag.eval import EvalResult
from autorag.loop import propose_next_config, run_research
from autorag.store import ExperimentStore
from autorag.usage import CallMetrics
from tests.helpers import sample_metrics


def _eval_result(cost_usd: float = 0.05, split: str = "dev") -> EvalResult:
    return EvalResult(
        metrics=sample_metrics(),
        split=split,
        n=7,
        cost=CostBreakdown(answering_usd=cost_usd),
    )


def _seed_baseline(store: ExperimentStore) -> None:
    metrics = sample_metrics()
    store.insert(
        config=PipelineConfig(),
        metrics=metrics,
        objective=compute_objective(metrics),
        status="kept",
        parent_id=None,
        cost_usd=0.01,
        is_baseline=True,
    )


def test_propose_next_config_skips_duplicate_hash(
    tmp_path: Path,
    test_settings: Settings,
) -> None:
    store = ExperimentStore(tmp_path / "experiments.db")
    _seed_baseline(store)
    seen_default = store.hash_config(PipelineConfig())
    duplicate = PipelineConfig()
    novel = PipelineConfig(chunk_size=500, chunk_overlap=50)

    responses = [
        json.dumps(duplicate.model_dump()),
        json.dumps(novel.model_dump()),
    ]

    class SequenceLLM:
        model_id = "stub"
        provider = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, **kwargs) -> str:
            raw = responses[self.calls]
            self.calls += 1
            return raw

    with patch("autorag.loop.LLMClient", return_value=SequenceLLM()):
        proposed = propose_next_config(
            settings=test_settings,
            program_md="test program",
            store=store,
            best=store.best(),
            usage=MagicMock(total_cost_usd=0.0),
        )

    assert store.hash_config(proposed) != seen_default
    assert proposed.chunk_size == 500
    store.close()


def test_run_research_stops_at_max_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    db_path = tmp_path / "experiments.db"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        vector_backend="faiss",
        max_experiments=2,
        max_usd_spend=100.0,
    )
    proposals = [
        PipelineConfig(chunk_size=500, chunk_overlap=50),
        PipelineConfig(chunk_size=520, chunk_overlap=50),
        PipelineConfig(chunk_size=540, chunk_overlap=50),
    ]

    def fake_propose(**kwargs):
        return proposals.pop(0)

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", side_effect=fake_propose),
        patch("autorag.loop.evaluate_config", return_value=_eval_result()),
        patch("autorag.loop.held_out_reserve_usd", return_value=0.0),
        patch("autorag.loop.estimate_researcher_cost_usd", return_value=0.0),
        patch("autorag.loop.estimate_eval_cost_usd", return_value=0.01),
        patch("autorag.loop._dev_eval_estimate", return_value=0.01),
    ):
        exit_code = run_research(
            settings,
            budget=2,
            store_path=db_path,
            fresh=True,
            report_held_out=False,
        )

    store = ExperimentStore(db_path)
    assert exit_code == 0
    assert store.count() == 2
    store.close()


def test_hard_cap_does_not_start_eval_that_would_overshoot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MAX_USD_SPEND", "0.10")
    db_path = tmp_path / "experiments.db"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        vector_backend="faiss",
        max_experiments=100,
        max_usd_spend=0.10,
    )
    store = ExperimentStore(db_path)
    _seed_baseline(store)
    store.close()

    eval_calls: list[PipelineConfig] = []

    def fake_evaluate(settings, config, *, split="dev", **kwargs):
        eval_calls.append(config)
        return _eval_result(cost_usd=0.06, split=split)

    def fake_propose(**kwargs):
        return PipelineConfig(chunk_size=500 + 10 * len(eval_calls), chunk_overlap=50)

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", side_effect=fake_propose),
        patch("autorag.loop.evaluate_config", side_effect=fake_evaluate),
        patch("autorag.loop.held_out_reserve_usd", return_value=0.0),
        patch("autorag.loop.estimate_researcher_cost_usd", return_value=0.0),
        patch("autorag.loop.estimate_eval_cost_usd", return_value=0.06),
        patch("autorag.loop._dev_eval_estimate", return_value=0.06),
    ):
        exit_code = run_research(
            settings,
            budget=100,
            store_path=db_path,
            fresh=False,
            report_held_out=False,
        )

    store = ExperimentStore(db_path)
    assert exit_code == 0
    assert len(eval_calls) == 1
    assert store.count() == 2
    assert store.total_cost_usd() == pytest.approx(0.07)
    store.close()


def test_eval_not_started_when_remaining_below_eval_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MAX_USD_SPEND", "0.10")
    db_path = tmp_path / "experiments.db"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        vector_backend="faiss",
        max_experiments=100,
        max_usd_spend=0.10,
    )
    store = ExperimentStore(db_path)
    _seed_baseline(store)
    store.close()

    eval_calls = 0
    propose_calls = 0

    def fake_propose(*, usage, **kwargs):
        nonlocal propose_calls
        propose_calls += 1
        usage.record(CallMetrics(cost_usd=0.01, operation="researcher"))
        return PipelineConfig(chunk_size=500 + propose_calls, chunk_overlap=50)

    def fake_evaluate(settings, config, *, split="dev", **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        return _eval_result(cost_usd=0.08, split=split)

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", side_effect=fake_propose),
        patch("autorag.loop.evaluate_config", side_effect=fake_evaluate),
        patch("autorag.loop.held_out_reserve_usd", return_value=0.0),
        patch("autorag.loop.estimate_researcher_cost_usd", return_value=0.01),
        patch("autorag.loop.estimate_eval_cost_usd", return_value=0.08),
        patch("autorag.loop._dev_eval_estimate", return_value=0.08),
    ):
        run_research(
            settings,
            budget=100,
            store_path=db_path,
            fresh=False,
            report_held_out=False,
        )

    assert eval_calls == 1
    assert propose_calls >= 1


def test_proposal_not_started_when_remaining_below_researcher_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MAX_USD_SPEND", "0.10")
    db_path = tmp_path / "experiments.db"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        max_experiments=100,
        max_usd_spend=0.10,
        vector_backend="faiss",
    )
    store = ExperimentStore(db_path)
    _seed_baseline(store)
    store.close()

    propose_calls = 0

    def fake_propose(**kwargs):
        nonlocal propose_calls
        propose_calls += 1
        return PipelineConfig(chunk_size=500, chunk_overlap=50)

    def fake_evaluate(settings, config, *, split="dev", **kwargs):
        return _eval_result(cost_usd=0.05, split=split)

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", side_effect=fake_propose),
        patch("autorag.loop.evaluate_config", side_effect=fake_evaluate),
        patch("autorag.loop.held_out_reserve_usd", return_value=0.0),
        patch("autorag.loop.estimate_researcher_cost_usd", return_value=0.06),
        patch("autorag.loop.estimate_eval_cost_usd", return_value=0.05),
        patch("autorag.loop._dev_eval_estimate", return_value=0.05),
    ):
        run_research(
            settings,
            budget=100,
            store_path=db_path,
            fresh=False,
            report_held_out=False,
        )

    assert propose_calls == 1


def test_held_out_eval_does_not_write_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    db_path = tmp_path / "experiments.db"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        vector_backend="faiss",
        max_experiments=1,
        max_usd_spend=10.0,
    )
    splits: list[str] = []

    def fake_evaluate(settings, config, *, split="dev", **kwargs):
        splits.append(split)
        return _eval_result(cost_usd=0.01, split=split)

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", return_value=PipelineConfig(chunk_size=500)),
        patch("autorag.loop.evaluate_config", side_effect=fake_evaluate),
        patch("autorag.loop.held_out_reserve_usd", return_value=0.02),
        patch("autorag.loop.estimate_researcher_cost_usd", return_value=0.0),
        patch("autorag.loop.estimate_eval_cost_usd", return_value=0.01),
    ):
        run_research(
            settings,
            budget=1,
            store_path=db_path,
            fresh=True,
            report_held_out=True,
        )

    store = ExperimentStore(db_path)
    assert store.count() == 1
    assert splits[0] == "dev"
    assert "test" in splits
    assert "adversarial" in splits
    store.close()
