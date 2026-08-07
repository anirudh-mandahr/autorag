"""Research loop budget caps and config dedup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorag.config import PipelineConfig, Settings, compute_objective
from autorag.loop import propose_next_config, run_research
from autorag.store import ExperimentStore

_SAMPLE_METRICS = {
    "answerable_recall": 0.9,
    "citation_rate": 0.8,
    "expected_source_hit_rate": 0.7,
    "groundedness": 0.85,
    "correct_refusal_rate": 1.0,
    "avg_cost_per_query": 0.002,
    "avg_latency": 100.0,
}


def _seed_baseline(store: ExperimentStore) -> None:
    metrics = dict(_SAMPLE_METRICS)
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
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *, temperature=0.0, response_format=None) -> str:
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
        patch("autorag.loop.run_experiment", return_value=(_SAMPLE_METRICS, 0.01)),
    ):
        exit_code = run_research(settings, budget=2, store_path=db_path, fresh=True)

    store = ExperimentStore(db_path)
    assert exit_code == 0
    # Baseline + exactly one proposed experiment (budget=2 session experiments total).
    assert store.count() == 2
    store.close()


def test_run_research_stops_at_max_usd_spend(
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

    # Pre-seed baseline so spend tracking only counts loop iterations.
    store = ExperimentStore(db_path)
    _seed_baseline(store)
    store.close()

    call_count = 0

    def fake_propose(**kwargs):
        return PipelineConfig(chunk_size=500 + call_count * 10, chunk_overlap=50)

    def fake_run_experiment(settings, config):
        nonlocal call_count
        call_count += 1
        return _SAMPLE_METRICS, 0.05

    with (
        patch("autorag.loop.load_program_md", return_value="program"),
        patch("autorag.loop.propose_next_config", side_effect=fake_propose),
        patch("autorag.loop.run_experiment", side_effect=fake_run_experiment),
    ):
        exit_code = run_research(settings, budget=100, store_path=db_path, fresh=False)

    store = ExperimentStore(db_path)
    assert exit_code == 0
    # Baseline (0.01) + two loop runs (0.05 each) => 0.11 >= 0.10 cap after 2nd loop iter.
    assert call_count == 2
    assert store.count() == 3
    assert store.total_cost_usd() == pytest.approx(0.11)
    store.close()
